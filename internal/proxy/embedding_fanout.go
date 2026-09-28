package proxy

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"sync"

	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/converter/vertex"
)

// embedContentFanOutConcurrency caps the parallel upstream calls one
// fanned-out embeddings request makes (Vertex AI embedContent embeds a single
// content per call).
const embedContentFanOutConcurrency = 8

// embeddingFanOutBodies returns conv's per-input embedContent bodies, or nil
// when the request goes upstream as one call (conv is nil on the native
// Responses path).
func embeddingFanOutBodies(conv *converter.ProviderConverter) [][]byte {
	if conv == nil {
		return nil
	}
	return conv.EmbeddingFanOutBodies()
}

type embedContentFanOutResult struct {
	resp *http.Response
	body []byte
	err  error
}

// doEmbedContentFanOut sends one Vertex AI embedContent call per body, cloning
// template (URL, headers, context — and with it the credential's egress proxy)
// for each, and answers as if it were a single upstream call:
//
//   - every call succeeded: a synthesized 200 whose body is the replies joined
//     in input order by vertex.MergeEmbedContentResponses;
//   - some call got a non-2xx reply: that reply (the lowest input index), so
//     retry, fail2ban and error mapping see the provider's own status and body;
//   - otherwise the first transport error.
//
// The first failure cancels the calls still pending: a partial set of vectors
// is useless to the client, and each extra call would still be billed upstream.
func (p *Proxy) doEmbedContentFanOut(template *http.Request, bodies [][]byte) (*http.Response, error) {
	ctx, cancel := context.WithCancel(template.Context())
	defer cancel()

	results := make([]embedContentFanOutResult, len(bodies))
	sem := make(chan struct{}, embedContentFanOutConcurrency)
	var wg sync.WaitGroup
	for i, body := range bodies {
		wg.Add(1)
		go func(i int, body []byte) {
			defer wg.Done()
			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
				results[i].err = ctx.Err()
				return
			}
			defer func() { <-sem }()

			req := template.Clone(ctx)
			req.Body = io.NopCloser(bytes.NewReader(body))
			req.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(body)), nil }
			req.ContentLength = int64(len(body))
			req.Header.Set("Content-Length", strconv.Itoa(len(body)))

			resp, err := p.client.Do(req) //nolint:gosec // G704: template's URL, host comes from the matched credential
			if err != nil {
				results[i].err = err
				cancel()
				return
			}
			data, readErr := p.readLimitedResponseBody(resp.Body)
			_ = resp.Body.Close()
			if readErr != nil {
				results[i].err = readErr
				cancel()
				return
			}
			results[i] = embedContentFanOutResult{resp: resp, body: data}
			if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
				cancel()
			}
		}(i, body)
	}
	wg.Wait()

	for _, result := range results {
		if result.resp != nil && (result.resp.StatusCode < http.StatusOK || result.resp.StatusCode >= http.StatusMultipleChoices) {
			return responseWithBody(result.resp, result.body), nil
		}
	}
	var cancelled error
	for _, result := range results {
		if result.err == nil {
			continue
		}
		// Calls we cancelled ourselves only report the failure that caused it.
		if errors.Is(result.err, context.Canceled) && template.Context().Err() == nil {
			cancelled = result.err
			continue
		}
		return nil, result.err
	}
	if cancelled != nil {
		return nil, cancelled
	}

	decoded := make([][]byte, len(results))
	for i, result := range results {
		decoded[i] = []byte(decodeResponseBody(result.body, result.resp.Header.Get("Content-Encoding")))
	}
	merged, err := vertex.MergeEmbedContentResponses(decoded)
	if err != nil {
		return nil, fmt.Errorf("merge embedContent fan-out responses: %w", err)
	}
	first := results[0].resp
	header := first.Header.Clone()
	header.Del("Content-Encoding")
	header.Del("Content-Length")
	header.Set("Content-Type", "application/json")
	return responseWithBody(&http.Response{
		Status:     first.Status,
		StatusCode: first.StatusCode,
		Proto:      first.Proto,
		ProtoMajor: first.ProtoMajor,
		ProtoMinor: first.ProtoMinor,
		Header:     header,
		Request:    template,
	}, merged), nil
}

// responseWithBody returns resp with body as its already-read, replayable Body.
func responseWithBody(resp *http.Response, body []byte) *http.Response {
	resp.Body = io.NopCloser(bytes.NewReader(body))
	resp.ContentLength = int64(len(body))
	return resp
}

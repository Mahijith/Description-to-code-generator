"""Isolates the live preview's `localStorage` per generation.

Streamlit's HTML-embedding components (`st.components.v1.html`, and its
replacement `st.iframe`) render the given HTML in a `srcdoc` iframe with
same-origin access to the Streamlit app itself — confirmed directly from
both components' own docstrings. Same origin means one shared
`localStorage` bucket: without this, generating a task tracker and then
regenerating a completely different app that also happens to use
`localStorage.setItem("tasks", ...)` (a very common key) would let the
second preview see, or silently corrupt, data left over from the first —
verified empirically (a throwaway Streamlit + Playwright probe run in this
sandbox), not just reasoned about.

This only wraps the copy of the code handed to the preview iframe. The
real, unmodified `result.code` (what the download button serves) is never
touched — a downloaded/deployed app should have real, permanent
`localStorage`, not a preview-only isolation shim.
"""

from __future__ import annotations


def isolate_local_storage(html: str, run_id: str) -> str:
    """Return `html` with a shim injected that namespaces every
    `localStorage` key under this specific `run_id`, so a fresh id (one
    per successful generation, including Regenerate) always starts from
    guaranteed-empty storage — matching what a user opening a freshly
    downloaded copy of this exact file would see.

    The shim is unconditionally prepended before the rest of the document
    (even before `<!doctype html>`) rather than inserted after a `<body`
    tag found by searching the text: a real generated app can easily
    contain that substring somewhere it doesn't mean the tag itself (a JS
    template string building HTML, a comment, a code sample) — confirmed
    directly by constructing exactly that case — silently corrupting the
    script it lands inside. A leading `<script>` still runs before any
    other inline script in document order (confirmed directly too), which
    is all this needs: it only has to override `localStorage` before the
    app's own code touches it, not sit in any particular element.
    """
    shim = f"""<script>
(function() {{
  var PREFIX = "gen_{run_id}__";
  var real = window.localStorage;
  var shim = {{
    getItem: function(k) {{ return real.getItem(PREFIX + k); }},
    setItem: function(k, v) {{ return real.setItem(PREFIX + k, v); }},
    removeItem: function(k) {{ return real.removeItem(PREFIX + k); }},
    clear: function() {{
      Object.keys(real).filter(function(k) {{ return k.indexOf(PREFIX) === 0; }})
        .forEach(function(k) {{ real.removeItem(k); }});
    }},
    key: function(i) {{
      var keys = Object.keys(real).filter(function(k) {{ return k.indexOf(PREFIX) === 0; }});
      return keys[i] ? keys[i].slice(PREFIX.length) : null;
    }},
  }};
  // Plain assignment (window.localStorage = shim) silently no-ops in
  // Chromium — confirmed directly. Object.defineProperty is required.
  Object.defineProperty(window, "localStorage", {{value: shim, configurable: true, writable: false}});
}})();
</script>"""
    return shim + html

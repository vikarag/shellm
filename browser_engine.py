"""Singleton browser engine using camoufox for stealthy web page fetching."""

import logging
import threading

logger = logging.getLogger(__name__)

# Camoufox — graceful degradation if not installed
try:
    from camoufox.sync_api import Camoufox
    CAMOUFOX_AVAILABLE = True
except ImportError:
    CAMOUFOX_AVAILABLE = False

MAX_TEXT_LENGTH = 50_000


class BrowserEngine:
    """Process-lifetime singleton browser via camoufox.

    First call to get_instance() launches a headless Firefox (~2s).
    Subsequent calls reuse the same browser. Each fetch_page() opens
    an isolated page that is closed after extraction.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._browser = None
        self._cm = None  # context manager reference for cleanup

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = cls()
                    inst._start()
                    cls._instance = inst
        return cls._instance

    @classmethod
    def is_available(cls):
        return CAMOUFOX_AVAILABLE

    def _start(self):
        """Launch the camoufox browser."""
        if not CAMOUFOX_AVAILABLE:
            raise RuntimeError("camoufox is not installed")
        self._cm = Camoufox(headless=True)
        self._browser = self._cm.__enter__()
        logger.info("BrowserEngine started (camoufox headless)")

    def fetch_page(self, url, timeout_ms=30000, wait_for_selector=None):
        """Fetch a URL and extract visible text.

        Args:
            url: The URL to fetch.
            timeout_ms: Navigation timeout in milliseconds.
            wait_for_selector: Optional CSS selector to wait for before extracting.

        Returns:
            dict with keys: text, title, url, error
        """
        if not self._browser:
            return {"text": "", "title": "", "url": url, "error": "Browser not started"}

        page = None
        try:
            page = self._browser.new_page()
            page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")

            if wait_for_selector:
                try:
                    page.wait_for_selector(wait_for_selector, timeout=min(timeout_ms, 10000))
                except Exception:
                    pass  # Continue with what we have

            title = page.title() or ""
            text = page.inner_text("body") or ""

            # Truncate to protect LLM context
            if len(text) > MAX_TEXT_LENGTH:
                text = text[:MAX_TEXT_LENGTH] + "\n\n[...truncated]"

            return {"text": text, "title": title, "url": page.url, "error": None}
        except Exception as e:
            return {"text": "", "title": "", "url": url, "error": str(e)}
        finally:
            if page:
                try:
                    page.close()
                except Exception:
                    pass

    def shutdown(self):
        """Close the browser and release resources."""
        if self._cm:
            try:
                self._cm.__exit__(None, None, None)
            except Exception:
                pass
            self._cm = None
            self._browser = None
        self.__class__._instance = None
        logger.info("BrowserEngine shut down")

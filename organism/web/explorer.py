from urllib.request import Request, urlopen
from urllib.parse import urlparse


class WebExplorer:
    def __init__(self, timeout=10):
        self.timeout = timeout

    def fetch(self, url):
        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            raise ValueError("Only HTTP and HTTPS URLs are allowed.")

        request = Request(
            url,
            headers={
                "User-Agent": "DIA-Research-Explorer/0.1"
            },
        )

        with urlopen(
            request,
            timeout=self.timeout,
        ) as response:
            content_type = response.headers.get(
                "Content-Type",
                "",
            )

            data = response.read(1000000)

            return {
                "url": url,
                "status": response.status,
                "content_type": content_type,
                "content": data.decode(
                    "utf-8",
                    errors="replace",
                ),
            }

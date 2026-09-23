import re
from html import unescape


class WebParser:
    def extract_text(self, html):
        html = re.sub(
            r"<script\b[^>]*>.*?</script>",
            " ",
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )

        html = re.sub(
            r"<style\b[^>]*>.*?</style>",
            " ",
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )

        text = re.sub(
            r"<[^>]+>",
            " ",
            html,
        )

        text = unescape(text)

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

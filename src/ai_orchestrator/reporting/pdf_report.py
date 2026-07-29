from datetime import datetime
from pathlib import Path

from ai_orchestrator.models import TestExecutionResult
from ai_orchestrator.models.execution_summary import ExecutionSummary
from ai_orchestrator.reporting.html_report import HtmlReport


class PdfReport:
    """
    Generates a PDF report by converting the HTML report via Playwright.

    Playwright manages its own headless Chromium browser, so no native
    system libraries (GTK, Cairo, Pango) are required on any platform.

    Requires:
        pip install playwright
        playwright install chromium
    """

    def __init__(self, output_dir: str = "reports/pdf"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._html_report = HtmlReport()

    def generate(
        self,
        summary: ExecutionSummary,
        execution_results: list[TestExecutionResult],
    ) -> Path:
        """
        Render HTML from the execution summary and convert it to PDF.

        Returns the path to the generated PDF file.
        Raises ImportError with a clear message if playwright is not installed
        or its browsers have not been downloaded.
        """
        try:
            from playwright.sync_api import sync_playwright  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "playwright is required for PDF generation.\n"
                "Install it with:\n"
                "  pip install playwright\n"
                "  playwright install chromium"
            ) from exc

        html_string = self._html_report.render(summary, execution_results)

        filename = f"execution_{datetime.now():%Y%m%d_%H%M%S}.pdf"
        output_file = self.output_dir / filename

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.set_content(html_string, wait_until="networkidle")
            page.pdf(
                path=str(output_file),
                format="A4",
                margin={"top": "20mm", "bottom": "20mm",
                        "left": "15mm", "right": "15mm"},
                print_background=True,
            )
            browser.close()

        return output_file

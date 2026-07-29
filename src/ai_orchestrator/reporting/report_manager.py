from ai_orchestrator.models import TestExecutionResult
from ai_orchestrator.reporting.summary import ExecutionSummaryBuilder
from ai_orchestrator.reporting.json_report import JsonReport
from ai_orchestrator.reporting.html_report import HtmlReport
from ai_orchestrator.reporting.pdf_report import PdfReport


class ReportManager:
    """
    Coordinates all report generators for a single execution run.

    The runner only needs to call ReportManager.generate() — it does not
    need to know about individual report formats or how summaries are built.

    Args:
        reports: list of report format names to generate, e.g.
            ["json", "html", "pdf"]. If empty/None, defaults to all formats.
    """

    _ALL_FORMATS = ("json", "html", "pdf")

    def __init__(self, reports: list[str] | None = None):
        self.reports = list(reports) if reports else list(self._ALL_FORMATS)

        self._json = JsonReport()
        self._html = HtmlReport()
        self._pdf = PdfReport()

    def generate(self, execution_results: list[TestExecutionResult]) -> None:
        """
        Build the execution summary and write the configured report formats.
        """
        summary = ExecutionSummaryBuilder.build(execution_results)

        print()
        print("=" * 60)

        if "json" in self.reports:
            json_path = self._json.generate(summary, execution_results)
            print(f"JSON report: {json_path}")

        if "html" in self.reports:
            html_path = self._html.generate(summary, execution_results)
            print(f"HTML report: {html_path}")

        if "pdf" in self.reports:
            try:
                pdf_path = self._pdf.generate(summary, execution_results)
                print(f"PDF  report: {pdf_path}")
            except ImportError as e:
                print(f"PDF  report: skipped — {e}")

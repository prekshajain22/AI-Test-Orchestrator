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
    """

    def __init__(self):
        self._json = JsonReport()
        self._html = HtmlReport()
        self._pdf = PdfReport()

    def generate(self, execution_results: list[TestExecutionResult]) -> None:
        """
        Build the execution summary and write JSON, HTML, and PDF reports.
        """
        summary = ExecutionSummaryBuilder.build(execution_results)

        json_path = self._json.generate(summary, execution_results)
        html_path = self._html.generate(summary, execution_results)

        print()
        print("=" * 60)
        print(f"JSON report: {json_path}")
        print(f"HTML report: {html_path}")

        try:
            pdf_path = self._pdf.generate(summary, execution_results)
            print(f"PDF  report: {pdf_path}")
        except ImportError as e:
            print(f"PDF  report: skipped — {e}")

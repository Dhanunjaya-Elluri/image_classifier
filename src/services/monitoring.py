"""Service for fetching monitoring metrics"""

import requests

from ..core.config import settings
from ..core.exceptions import PrometheusConnectionError


class MonitoringService:
    """Service for fetching monitoring metrics"""

    def __init__(self):
        self.prometheus_url = settings.PROMETHEUS_URL
        self.timeout = 5

    async def get_metrics(self):
        """Fetch metrics from Prometheus"""
        try:
            # Get total predictions
            total_query = "sum(image_classifier_predictions_total)"
            total_response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": total_query},
                timeout=self.timeout,
            )
            total_response.raise_for_status()
            total_data = total_response.json()

            # Get success rate for predictions
            success_query = (
                'sum(image_classifier_predictions_total{status="200"}) / '
                "sum(image_classifier_predictions_total) * 100"
            )
            success_response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": success_query},
                timeout=self.timeout,
            )
            success_response.raise_for_status()
            success_data = success_response.json()

            # Get histogram bucket counts
            histogram_query = "sum(image_classifier_prediction_seconds_bucket) by (le)"
            histogram_response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": histogram_query},
                timeout=self.timeout,
            )
            histogram_response.raise_for_status()
            histogram_data = histogram_response.json()

            # Process the results with safe value handling
            total_predictions = max(
                0,
                float(
                    total_data["data"]["result"][0]["value"][1]
                    if total_data["data"]["result"]
                    else 0
                ),
            )

            success_rate = min(
                100,
                max(
                    0,
                    float(
                        success_data["data"]["result"][0]["value"][1]
                        if success_data["data"]["result"]
                        else 100
                    ),
                ),
            )

            # Process histogram data
            histogram_buckets = []
            if histogram_data["data"]["result"]:
                results = histogram_data["data"]["result"]
                # Sort by bucket threshold
                results.sort(key=lambda x: float(x["metric"]["le"]))

                prev_count = 0
                for result in results:
                    bucket = result["metric"].get("le", "inf")
                    if bucket != "inf":
                        current_count = float(result["value"][1])
                        # Calculate the actual count in this bucket
                        bucket_count = current_count - prev_count
                        histogram_buckets.append(
                            {"bucket": float(bucket), "count": bucket_count}
                        )
                        prev_count = current_count

            return {
                "requests": {
                    "total": total_predictions,
                    "success": (success_rate / 100) * total_predictions,
                    "error": ((100 - success_rate) / 100) * total_predictions,
                },
                "response_times": histogram_buckets if total_predictions > 0 else [],
            }

        except requests.exceptions.RequestException as e:
            raise PrometheusConnectionError(
                "Unable to connect to Prometheus server"
            ) from e

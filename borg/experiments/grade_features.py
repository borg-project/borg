"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import csv
import borg

logger = borg.get_logger(__name__, default_level = "INFO")

@borg.annotations(
    out_path = ("results CSV output path",),
    bundle_path = ("path to run data",)
    )
def main(out_path, bundle_path):
    """Grade the utility of instance features."""

    run_data = borg.storage.RunData.from_bundle(bundle_path)
    model = borg.models.MulEstimator()(run_data, 60, run_data)
    regress = borg.regression.NearestRTDRegression(model)
    names = sorted(run_data.common_features)
    weights = regress.classifier.get_feature_weights()

    with borg.util.openz(out_path, "wb") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["name", "weight"])
        writer.writerows(zip(names, weights))

        out_file.flush()

if __name__ == "__main__":
    borg.script(main)


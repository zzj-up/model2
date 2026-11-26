"""Only audited reporting-only trainer upgrades can resume prior optimizer state."""


def reports_only_upgrade(saved_sha, current_sha):
    """Both exact versions must match; later code changes require another review."""
    return (saved_sha, current_sha) == (
        "54e55323a0459a707ee667c88c78409bca3db7ddca17f82c2ba0e9261632c5ed",
        "e3f0ec23a451d1322456a2c27c5b6efa1a4efe93b590fb9e404036845ae43066",
    )

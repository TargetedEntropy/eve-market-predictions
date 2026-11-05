# Operator-reviewed prerequisite, not applied by deployment jobs.
# Install checksum-verified hashicorp/nomad-device-nvidia release in plugin_dir.
plugin_dir = "/opt/nomad/plugins"
plugin "nvidia-gpu" {
  config {
    enabled = true
    fingerprint_period = "1m"
  }
}
plugin "docker" {
  config {
    allow_privileged = false
    nvidia_runtime = "nvidia"
  }
}

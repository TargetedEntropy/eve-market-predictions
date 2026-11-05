variable "namespace" { type = string }
variable "datacenter" { type = string }
variable "worker_image" { type = string }
variable "artifact_volume_id" { type = string }

job "jita-gpu" {
  namespace = var.namespace
  datacenters = [var.datacenter]
  type = "batch"
  periodic {
    crons = ["15 * * * *"]
    time_zone = "UTC"
    prohibit_overlap = true
  }
  group "batch" {
    count = 1
    volume "artifacts" {
      type = "csi"
      source = var.artifact_volume_id
      read_only = false
      attachment_mode = "file-system"
      access_mode = "multi-node-multi-writer"
    }
    restart {
      attempts = 0
      mode = "fail"
    }
    reschedule {
      attempts = 0
      unlimited = false
    }
    task "forecast" {
      driver = "docker"
      user = "10001:10001"
      config {
        image = var.worker_image
        readonly_rootfs = true
        tmpfs = ["/tmp:rw,nosuid,nodev,size=1g,mode=1777"]
        cap_drop = ["ALL"]
        security_opt = ["no-new-privileges"]
      }
      env {
        JITA_ARTIFACT_DIR = "/artifacts"
        NVIDIA_DRIVER_CAPABILITIES = "compute,utility"
        OMP_NUM_THREADS = "4"
        MKL_NUM_THREADS = "4"
        CUDA_CACHE_PATH = "/tmp/cuda-cache"
      }
      template {
        destination = "secrets/source.env"
        env = true
        perms = "0400"
        change_mode = "restart"
        data = <<EOH
{{ with nomadVar "nomad/jobs/jita-gpu" }}
JITA_SOURCE_URL={{ .source_url | toJSON }}
JITA_SOURCE_TOKEN={{ .source_token | toJSON }}
{{ end }}
EOH
      }
      volume_mount {
        volume = "artifacts"
        destination = "/artifacts"
        read_only = false
      }
      resources {
        cpu = 4000
        memory = 8192
        device "nvidia/gpu" {
          count = 1
          constraint {
            attribute = "${device.model}"
            operator = "regexp"
            value = ".*RTX 5080.*"
          }
        }
      }
      kill_timeout = "60s"
      logs {
        max_files = 5
        max_file_size = 20
      }
    }
  }
}

variable "namespace" { type = string }
variable "datacenter" { type = string }
variable "api_image" { type = string }
variable "artifact_volume_id" { type = string }

job "jita-api" {
  namespace = var.namespace
  datacenters = [var.datacenter]
  type = "service"
  group "api" {
    count = 1
    network {
      port "http" { to = 8000 }
    }
    volume "artifacts" {
      type = "csi"
      source = var.artifact_volume_id
      read_only = true
      attachment_mode = "file-system"
      access_mode = "multi-node-multi-writer"
    }
    update {
      max_parallel = 1
      min_healthy_time = "10s"
      healthy_deadline = "3m"
      auto_revert = true
    }
    restart {
      attempts = 3
      interval = "5m"
      delay = "10s"
      mode = "fail"
    }
    task "api" {
      driver = "docker"
      user = "10001:10001"
      config {
        image = var.api_image
        ports = ["http"]
        readonly_rootfs = true
        tmpfs = ["/tmp:rw,nosuid,nodev,size=64m,mode=1777"]
        cap_drop = ["ALL"]
        security_opt = ["no-new-privileges"]
      }
      env {
        JITA_ARTIFACT_DIR = "/artifacts"
        JITA_STALE_SECONDS = "10800"
        JITA_REQUESTS_PER_MINUTE = "240"
      }
      template {
        destination = "secrets/api.env"
        env = true
        perms = "0400"
        change_mode = "restart"
        data = <<EOH
{{ with nomadVar "nomad/jobs/jita-api" }}
JITA_API_TOKEN={{ .api_token | toJSON }}
{{ end }}
EOH
      }
      volume_mount {
        volume = "artifacts"
        destination = "/artifacts"
        read_only = true
      }
      resources {
        cpu = 500
        memory = 512
      }
      service {
        provider = "nomad"
        name = "jita-predictions"
        port = "http"
        tags = ["private", "jita", "cpu"]
        check {
          name = "liveness"
          type = "http"
          path = "/v1/health"
          interval = "15s"
          timeout = "3s"
        }
      }
      logs {
        max_files = 3
        max_file_size = 10
      }
    }
  }
}

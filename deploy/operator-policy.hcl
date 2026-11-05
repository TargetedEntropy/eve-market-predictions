# Change namespace to the provisioned namespace before applying.
namespace "jita" {
  policy = "write"
  capabilities = ["csi-mount-volume", "csi-list-volume", "read-job", "list-jobs"]
}
# Workload identities receive only their matching nomad/jobs/<job> variable
# through Nomad's implicit workload policy, not this human operator token.

# Create a CPU pod and wait until the container is up (no ssh)

## Prompt

Create a CPU pod from `ubuntu:22.04` for a quick preprocessing job and tell me when it's
up. I don't need to ssh into it.

## Expected behavior

The agent should:

1. Create with `runpodctl pod create --compute-type cpu --image ubuntu:22.04 --wait-for running`
   (runpodctl v2.15.0+), which returns once the container is reported up and needs no ssh.
2. NOT use plain `--wait` (ssh mode): `ubuntu:22.04` starts no sshd and cpu pods get no
   Runpod-managed ssh, so it would bill through the whole timeout and then fail.
3. NOT tell the user the pod is ssh-reachable — running mode proves nothing about ssh.
4. NOT write its own poll loop around `pod get`.
5. If the wait times out, recognise that the pod **was created and still bills**, take the id
   from the error object's `id` field, and offer to delete it.

The ssh scenario stays in `create-and-wait-until-usable.eval.md`: when the user needs ssh,
plain `--wait` remains correct, including for a cpu image that starts its own sshd.

## Assertions

- Uses `--wait-for running` (optionally with `--wait-timeout`), not plain `--wait` or a hand-rolled polling loop
- Does NOT include `--gpu-id` or `--gpu-count`
- Does not claim or imply the pod accepts ssh
- On `wait_timeout` / `wait_interrupted`, surfaces the pod id from the error object and proposes cleanup instead of assuming nothing was created

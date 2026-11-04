# Authentication and private deployment data

The source package contains no robot driver, device address, calibration,
operator account, SSH tunnel, login cache or supplied API key. The offline
example needs none of them. Real integrations supply their own device and
model factories outside the core.

## Optional Codex provider

The reference provider starts a locally installed Codex app-server only when a
turn is requested. It checks the deployer's existing ChatGPT login through
`account/read`, discards the account payload, and does not call a login flow or
copy an authentication file. This reference client currently requires ChatGPT
authentication; API-key-only and other model services need a different session
factory with the same interface.

Codex itself manages authentication outside this repository. Its
[official authentication documentation](https://learn.chatgpt.com/docs/auth)
describes local `auth.json`/OS credential storage. Keep that state outside the
repository. Deployers select the account and model access themselves; neither
is distributed with the project.

Device credentials belong in the device backend's environment, OS credential
store or secret manager. Provider credentials belong to that provider. Use
settings for non-secret identifiers and configuration, rather than placing
passwords or tokens in a task, Skill, profile, observation or example JSON.
The runner does not save a wholesale copy of factory settings.

## Runtime evidence

New runs contain the supplied images, state, actions, programs and knowledge.
These are private deployment data. The `.gitignore` excludes common credentials,
run outputs and knowledge roots; it cannot protect a secret placed elsewhere.
Diagnostic errors redact common credential keys, authentication headers and URL
credentials. This is best-effort text redaction, not an anonymizer for arbitrary
images, device state or unlabelled values. Review any material before publishing
it, as was done for this repository's selected demonstration videos.

The provider's empty working directory is not a filesystem read sandbox.
The Codex execution-feature restrictions are defense in depth, not a guarantee
that every possible provider tool is inaccessible. Choose appropriate OS and
provider isolation for your deployment. Changing authentication or factories
does not grant new device capabilities.

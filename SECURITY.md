# Security Policy

## Supported versions

Only the latest commit on `main` is supported. There are no release branches.

## Reporting a vulnerability

Please do **not** open a public issue for security problems.

Report privately through GitHub: open the **Security** tab of this repository and choose **Report a vulnerability**. Include:

- what you found and where (file, function, request),
- steps or a minimal payload to reproduce it,
- the impact you expect.

You should get an acknowledgement within 5 working days. Confirmed issues are fixed on `main` with a regression test, and the reporter is credited unless they ask otherwise.

## Scope

tolerance-travel is an evaluation harness using synthetic data. The optional live-model mode reads `ANTHROPIC_API_KEY` from the environment only; never commit keys or put them in config files. Live runs incur API costs.

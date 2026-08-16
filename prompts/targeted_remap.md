# Literature Multiverse targeted remap

Prompt version: `targeted-remap-v1`

Extract only the single human-approved context field defined below. Return exactly the supplied
strict schema. Echo each provided `finding_id` once; never create, rewrite, merge, or omit an ID.
Use null when the supplied paper context does not support a value. Do not infer values from the
effect direction or outcome.

Approved field definition:

`[[APPROVED_FIELD_JSON]]`

Finding contexts:

`[[FINDING_CONTEXTS_JSON]]`

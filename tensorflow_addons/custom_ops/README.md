# Addons - Custom Ops

## Contents
| Layer  | Description                             |
|:----------------------- |:-----------------------------|
| Image | Ops for image manipulation   |
| Text |  Ops for text processing  |



## Contribution Guidelines
#### Standard API
In order to conform with the current API standard, all custom ops
must:
 * Must be impossible to implement in one of the other API
 standards (Layers, Losses, etc.)

#### Testing Requirements
 * Simple unittests that demonstrate the custom op is behaving as
    expected.
 * When applicable, run all unittests with TensorFlow's
  `@run_all_in_graph_and_eager_modes` decorator.

#### Documentation Requirements
 * Update the table of contents in the project's central README
 * Update the table of contents in this sub-project's README
 
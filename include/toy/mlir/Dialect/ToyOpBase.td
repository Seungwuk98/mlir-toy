#ifndef TOY_OP_BASE_TD
#define TOY_OP_BASE_TD

include "mlir/IR/OpBase.td"
include "toy/mlir/Dialect/ToyDialect.td"

// Base class for toy dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Toy_Op<string mnemonic, list<Trait> traits = []>
    : Op<Toy_Dialect, mnemonic, traits>;

#endif // TOY_OP_BASE_TD

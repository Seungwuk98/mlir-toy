#ifndef TOY_SHAPE_INFERENCE_H
#define TOY_SHAPE_INFERENCE_H

#include "mlir/Pass/Pass.h"

namespace mlir::toy {

std::unique_ptr<mlir::Pass> createShapeInferencePass();

std::unique_ptr<mlir::Pass> createToyToAffineLoweringPass();

std::unique_ptr<mlir::Pass> createAffineToLLVMLoweringPass();

std::unique_ptr<mlir::Pass> createPrintOpLoweringPass();

void registerShapeInferencePass();

void registerToyToAffineLoweringPass();

void registerAffineToLLVMLoweringPass();

void registerPrintOpLoweringPass();

} // namespace mlir::toy

#endif // TOY_SHAPE_INFERENCE_H

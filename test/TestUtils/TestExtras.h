#ifndef TOY_TEST_EXTRAS_H
#define TOY_TEST_EXTRAS_H

#include "doctest/doctest.h"
#include "llvm/ADT/StringRef.h"
#include <format>

#define STR_EQ(E, R) CHECK(::toy::test::StringCompare(E, R))

namespace toy::test {

inline bool StringCompare(llvm::StringRef E, llvm::StringRef R) {
  E = E.trim();
  R = R.trim();

  if (E != R) {
    FAIL(std::format("Expected:\n{}\n\nReceived:\n{}\n", E.str(), R.str()));
    return false;
  }
  return true;
}

} // namespace toy::test

#endif // TOY_TEST_EXTRAS_H

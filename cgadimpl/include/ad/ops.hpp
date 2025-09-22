// =====================
// file: include/ag/ops.hpp (declarations only)
// =====================
#pragma once
#include "ad/graph.hpp"


namespace ag {


Value add (const Value& a, const Value& b);
Value sub (const Value& a, const Value& b);
Value mul (const Value& a, const Value& b);
Value relu (const Value& x);
Value matmul(const Value& a, const Value& b);
Value sum (const Value& x);


inline Value operator+(const Value& a, const Value& b){ return add(a,b);}
inline Value operator-(const Value& a, const Value& b){ return sub(a,b);}
inline Value operator*(const Value& a, const Value& b){ return mul(a,b);}


} // namespace ag
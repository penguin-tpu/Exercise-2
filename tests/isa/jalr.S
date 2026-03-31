.globl _start
_start:
  auipc t0, 0
  addi t0, t0, 20
  jalr x0, 0(t0)
  addi a0, x0, 1
  ebreak
target:
  addi a0, x0, 18
  ebreak

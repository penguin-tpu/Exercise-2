.globl _start
_start:
  auipc t0, 0
  auipc t1, 0
  addi t0, t0, 4
  sub a0, t1, t0
  addi a0, a0, 7
  ebreak

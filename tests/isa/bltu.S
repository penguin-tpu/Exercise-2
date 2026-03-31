.globl _start
_start:
  addi t0, x0, 1
  addi t1, x0, -1
  bltu t0, t1, first_ok
  addi a0, x0, 1
  ebreak
first_ok:
  bltu t1, t0, fail
  addi a0, x0, 15
  ebreak
fail:
  addi a0, x0, 2
  ebreak

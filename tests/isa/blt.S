.globl _start
_start:
  addi t0, x0, -1
  addi t1, x0, 1
  blt t0, t1, first_ok
  addi a0, x0, 1
  ebreak
first_ok:
  blt t1, t0, fail
  addi a0, x0, 13
  ebreak
fail:
  addi a0, x0, 2
  ebreak

.globl _start
_start:
  addi t0, x0, 3
  addi t1, x0, 4
  bne t0, t1, first_ok
  addi a0, x0, 1
  ebreak
first_ok:
  bne t0, t0, fail
  addi a0, x0, 12
  ebreak
fail:
  addi a0, x0, 2
  ebreak

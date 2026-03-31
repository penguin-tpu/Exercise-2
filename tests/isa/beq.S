.globl _start
_start:
  addi t0, x0, 5
  addi t1, x0, 5
  beq t0, t1, first_ok
  addi a0, x0, 1
  ebreak
first_ok:
  addi t1, t1, 1
  beq t0, t1, fail
  addi a0, x0, 11
  ebreak
fail:
  addi a0, x0, 2
  ebreak

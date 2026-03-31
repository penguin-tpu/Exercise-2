.globl _start
_start:
  lui t0, 0x20000
  addi t1, x0, 0x34
  addi t2, x0, 0x12
  slli t2, t2, 8
  or t1, t1, t2
  sh t1, 0(t0)
  lh a0, 0(t0)
  ebreak

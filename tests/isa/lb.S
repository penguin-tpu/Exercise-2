.globl _start
_start:
  lui t0, 0x20000
  addi t1, x0, -1
  sb t1, 0(t0)
  lb a0, 0(t0)
  addi a0, a0, 2
  ebreak

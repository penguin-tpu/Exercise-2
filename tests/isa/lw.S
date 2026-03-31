.globl _start
_start:
  lui t0, 0x20000
  addi t1, x0, -128
  sb t1, 0(t0)
  lbu t2, 0(t0)
  lb t3, 0(t0)
  addi t3, t3, 256
  addi t4, x0, 0x34
  addi t5, x0, 0x12
  slli t5, t5, 8
  or t4, t4, t5
  sh t4, 2(t0)
  lhu t6, 2(t0)
  add a0, t2, t3
  add a0, a0, t6
  sw a0, 4(t0)
  lw a0, 4(t0)
  ebreak

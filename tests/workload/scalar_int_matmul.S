.section .data
A:
  .word 1, 2, 3, 4
B:
  .word 5, 6, 7, 8

.section .text
.globl _start
_start:
  la s0, A
  la s1, B
  lui s2, 0x20000

  lw t0, 0(s0)
  lw t1, 0(s1)
  jal ra, mul_pos
  add t6, x0, a0
  lw t0, 4(s0)
  lw t1, 8(s1)
  jal ra, mul_pos
  add t6, t6, a0
  sw t6, 0(s2)

  lw t0, 0(s0)
  lw t1, 4(s1)
  jal ra, mul_pos
  add t6, x0, a0
  lw t0, 4(s0)
  lw t1, 12(s1)
  jal ra, mul_pos
  add t6, t6, a0
  sw t6, 4(s2)

  lw t0, 8(s0)
  lw t1, 0(s1)
  jal ra, mul_pos
  add t6, x0, a0
  lw t0, 12(s0)
  lw t1, 8(s1)
  jal ra, mul_pos
  add t6, t6, a0
  sw t6, 8(s2)

  lw t0, 8(s0)
  lw t1, 4(s1)
  jal ra, mul_pos
  add t6, x0, a0
  lw t0, 12(s0)
  lw t1, 12(s1)
  jal ra, mul_pos
  add t6, t6, a0
  sw t6, 12(s2)

  lw a0, 12(s2)
  ebreak

mul_pos:
  addi t2, x0, 0
mul_loop:
  beq t1, x0, mul_done
  add t2, t2, t0
  addi t1, t1, -1
  jal x0, mul_loop
mul_done:
  addi a0, t2, 0
  jalr x0, 0(ra)

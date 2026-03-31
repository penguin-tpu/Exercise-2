.globl _start
_start:
  addi t0, x0, 5
  addi t1, x0, 0
loop:
  add t1, t1, t0
  addi t0, t0, -1
  bne t0, x0, loop
  addi t2, x0, -1
  addi t3, x0, 1
  blt t2, t3, signed_ok
  addi a0, x0, 1
  ebreak
signed_ok:
  bgeu t2, t3, unsigned_ok
  addi a0, x0, 2
  ebreak
unsigned_ok:
  bltu t3, t2, ordered_ok
  addi a0, x0, 3
  ebreak
ordered_ok:
  bge t3, x0, done
  addi a0, x0, 4
  ebreak
done:
  addi a0, t1, 0
  ebreak

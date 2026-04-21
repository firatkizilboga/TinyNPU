.section .text.start
.global _start
.type _start, @function

.equ STDOUT_REG, 0x10000000
.equ EXIT_REG,   0x20000004

_start:
    /* basic progress mark */
    li t0, STDOUT_REG
    li t1, 'S'
    sw t1, 0(t0)

    /* enable floating-point state: mstatus.FS = Dirty */
    li t0, (3 << 13)
    csrs mstatus, t0

    /* fa0 = 2.0f */
    li t0, 0x40000000
    fmv.w.x fa0, t0

    /* fa1 = sqrt(2.0f) */
    fsqrt.s fa1, fa0

    /* move raw bits back to integer register */
    fmv.x.w t1, fa1

    /* compare against expected single-precision bits */
    li t2, 0x3fb504f3
    li t3, EXIT_REG
    beq t1, t2, 1f

    /* mismatch */
    li t0, STDOUT_REG
    li t4, 'F'
    sw t4, 0(t0)
    li t4, 1
    sw t4, 0(t3)
    wfi

1:
    li t0, STDOUT_REG
    li t4, 'P'
    sw t4, 0(t0)
    sw zero, 0(t3)
    wfi

.size _start, .-_start

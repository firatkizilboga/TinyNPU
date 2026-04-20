#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

volatile uint32_t runtime_cycle_start __attribute__((section(".noinit")));
volatile uint32_t runtime_cycle_post_bss __attribute__((section(".noinit")));
volatile uint32_t runtime_cycle_post_init __attribute__((section(".noinit")));
volatile uint32_t runtime_cycle_pre_main __attribute__((section(".noinit")));

int main(void)
{
    const float sqrt_inputs[] = {0.5f, 1.0f, 2.0f, 4.0f, 9.0f};
    const float exp_inputs[] = {-4.0f, -1.0f, 0.0f, 1.0f, 4.0f};
    const float tanh_inputs[] = {-4.0f, -1.0f, 0.0f, 1.0f, 4.0f};

    printf("float_math_probe\n");
    for (int i = 0; i < 5; ++i) {
        printf("sqrtf(%f)=%f\n", (double)sqrt_inputs[i], (double)sqrtf(sqrt_inputs[i]));
    }
    for (int i = 0; i < 5; ++i) {
        printf("expf(%f)=%f\n", (double)exp_inputs[i], (double)expf(exp_inputs[i]));
    }
    for (int i = 0; i < 5; ++i) {
        printf("tanhf(%f)=%f\n", (double)tanh_inputs[i], (double)tanhf(tanh_inputs[i]));
    }
    printf("EXIT SUCCESS\n");
    return EXIT_SUCCESS;
}

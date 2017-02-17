
mot_float_type SSFP(
        const mot_float_type d,
        const mot_float_type delta,
        const mot_float_type G,
        const mot_float_type TR,
        const mot_float_type flip_angle,
        const mot_float_type b1,
        const mot_float_type T1,
        const mot_float_type T2){

    mot_float_type cos_b1_corrected_flip_angle;
    const mot_float_type sin_b1_corrected_flip_angle = sincos(flip_angle * b1, &cos_b1_corrected_flip_angle);

    const mot_float_type E1 = exp(-TR / T1);
    const mot_float_type E2 = exp(-TR / T2);

    const mot_float_type q_magnitude_2 = GAMMA_H_SQ * (G * G) * (delta * delta);

    const mot_float_type b = q_magnitude_2 * TR;
    const mot_float_type beta = q_magnitude_2 * delta;

    const mot_float_type A1 = exp(-b * d);
    const mot_float_type A2 = exp(-beta * d);

    const mot_float_type s = E2 * A1 * pow(A2, -4/3.0) * (1 - (E1 * cos_b1_corrected_flip_angle)) + E2 * pow(A2, -1/3.0) * (cos_b1_corrected_flip_angle - 1);

    const mot_float_type r = 1 - E1 * cos_b1_corrected_flip_angle + pown(E2, 2) * A1 * pow(A2, 1/3.0) * (cos_b1_corrected_flip_angle - E1);

    const mot_float_type K = (1 - E1 * A1 * cos_b1_corrected_flip_angle - pown(E2, 2) * pown(A1, 2) * pow(A2, -2/3.0) * (E1 * A1 - cos_b1_corrected_flip_angle))
                                /(E2 * A1 * pow(A2, -4/3.0) * (1 + cos_b1_corrected_flip_angle) * (1 - E1 * A1));

    const mot_float_type F1 = K - sqrt(pown(K, 2) - pown(A2, 2));

    return -((1 - E1) * E2 * pow(A2, -2/3.0) * (F1 - E2 * A1 * pow(A2, 2/3.0)) * sin_b1_corrected_flip_angle)/(r - F1*s);
}

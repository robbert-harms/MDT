#ifndef DMRICM_GDRCYLINDERS_CL
#define DMRICM_GDRCYLINDERS_CL

/**
 * Author = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

/** Small number constant used in continued fraction gamma evaluation */
#define GDRCYL_FPMIN 1E-30

/** Small number constant used in gamma series evaluation */
#define GDRCYL_EPS 3E-7

/** Max number of iterations in series evaluation */
#define GDRCYL_ITMAX 100

double gammaCDF(const double k, const double theta, const double x);
double gammp(const double a, const double x);
double gser(const double a, const double x);
double gcf(const double a, const double x);
double findGammaCDFCrossing(double startx, double stopx, const double offset, const double convergence,
                            const double gamma_k, const double gamma_beta);

model_float cmGDRCylinders(const double4 g,
                      const double G,
                      const double Delta,
                      const double delta,
                      const double d,
                      const double theta,
                      const double phi,
                      const double gamma_k,
                      const double gamma_beta,
                      const double gamma_nmr_cyl,
                      global const double* const CLJnpZeros,
                      const int CLJnpZerosLength){

    int nmr_cyl = round(gamma_nmr_cyl);

    double lower = findGammaCDFCrossing(0, gamma_beta*gamma_k, 1.0/nmr_cyl, 1e-20, gamma_k, gamma_beta);
    double upper = findGammaCDFCrossing(lower, nmr_cyl*gamma_beta*gamma_k, (1-1.0/nmr_cyl), 1e-20,
                                        gamma_k, gamma_beta);

    double binWidth = (upper-lower)/nmr_cyl;
    double gamma_cyl_weight = 0;
    double gamma_cyl_radius = 0;
    double signal = 0;

    for(int i = 0; i < nmr_cyl; i++){
        gamma_cyl_radius = lower + (i+0.5)*binWidth;
        gamma_cyl_weight = (gammaCDF(gamma_k, gamma_beta, lower + (i+1)*binWidth)
                                - gammaCDF(gamma_k, gamma_beta, lower + i*binWidth))
                                    / (1 - (2.0/nmr_cyl));

        signal += gamma_cyl_weight * cmCylinderGPD(g, G, Delta, delta, d, theta, phi,
                                                   gamma_cyl_radius, CLJnpZeros, CLJnpZerosLength);
    }
    return signal;
}

/**
 * Taken from Camino
 * Calculates the cumulative Gamma function up to the value given
 *
 * @param k gamma shape param
 * @param theta gamma scale param
 * @param x top end upper limit of integral
 *
 * @return gamma(x/theta)/Gamma(k)
 *         gamma(k, z)= incomplete gamma fn
 *         Gamma(k)= gamma function
 *
 *
 */
double gammaCDF(const double k, const double theta, const double x){
    return gammp(k, x/theta);
}

/**
 * Taken from Camino
 * Returns the incomplete gamma function P(a; x).
 * see NRC p. 218.
 */
double gammp(const double a, const double x){
    if(x<0.0 || a <= 0.0){
        return NAN;
    }

    if(x < (a + 1.0)){
        return gser(a, x);
    }

    return 1.0 - gcf(a, x);
}

/**
 * Returns the incomplete gamma function P(a; x) evaluated by its
 * series representation as gamser.
 */
double gser(const double a, const double x){
    double sum;
    double del;
    double ap;

    if(x <= 0.0){
        if (x < 0.0){
            return NAN;
        }
        return 0.0;
    }
    else{
        ap=a;
        del = sum = 1.0 / a;

        for(int n = 1; n <= GDRCYL_ITMAX; n++){
            ++ap;
            del *= x/ap;
            sum += del;

            if(fabs(del) < fabs(sum) * GDRCYL_EPS){
                return sum*exp(-x + a * log(x) - lgamma(a));
            }
        }
    }
    return NAN;
}

/*
 * Returns the incomplete gamma function Q(a; x) evaluated by its continued
 * fraction representation.
 */
double gcf(const double a, const double x){
    int i;
    double an,b,c,d,del,h;

    //Set up for evaluating continued fraction by modified Lentz's method (x5.2) with b0 = 0.
    b=x+1.0-a;
    c=1.0/GDRCYL_FPMIN;
    d=1.0/b;
    h=d;
    for(i=1; i<=GDRCYL_ITMAX; i++){
        an = -i*(i-a);
        b += 2.0;
        d=an*d+b;

        if(fabs(d) < GDRCYL_FPMIN){
            d=GDRCYL_FPMIN;
        }

        c=b+an/c;

        if(fabs(c) < GDRCYL_FPMIN){
            c=GDRCYL_FPMIN;
        }

        d=1.0/d;
        del=d*c;
        h *= del;

        if(fabs(del-1.0) < GDRCYL_EPS){
            break;
        }
    }
    if(i > GDRCYL_ITMAX)
        return NAN;

    return exp(-x+a*log(x)-lgamma(a))*h;
}

//Using Brent root finding to determine cdfs
double findGammaCDFCrossing(double startx, double stopx, const double offset, const double convergence,
                            const double gamma_k, const double gamma_beta){

    double fstartx = gammaCDF(gamma_k, gamma_beta, startx) - offset;
    double fstopx = gammaCDF(gamma_k, gamma_beta, stopx) - offset;
    double delta = fabs(stopx-startx);

    if(fstartx * fstopx > 0){
        if (fstartx>0){
            fstartx = gammaCDF(gamma_k, gamma_beta, 0) - offset;
        }
        else if (fstopx<0){
            fstopx = gammaCDF(gamma_k, gamma_beta, stopx/gamma_k) - offset;
        }
        else{
            return NAN;
        }
    }

    double root = startx;
    double froot = fstartx;
    bool mflag=1;
    double s = 0;
    double de = 0;

    while(!(delta < convergence || fstartx == 0 || fstopx == 0)){
         if (fstartx != froot && fstopx != froot){
             //inverse interpolation
             s = startx * fstopx * froot / ((fstartx-fstopx)*(fstartx-froot));
             s += stopx * fstartx * froot / ((fstopx-fstartx)*(fstopx-froot));
             s += root * fstartx * fstopx / ((froot-fstartx)*(froot-stopx));
         }
         else{
             //secant method
             s = stopx - fstopx * (stopx-startx) / (fstopx-fstartx);
         }

         //bisection
         if( !((s >= (3*startx+stopx)/4) && (s<=stopx) || (s<=(3*startx+stopx)/4) && (s>stopx))
                || mflag && (fabs(s-stopx) >= fabs(stopx-root)/2)
                || !mflag && (fabs(s-stopx) >= fabs(root-de)/2)
                || mflag && (fabs(stopx-root) < delta)
                || !mflag && (fabs(root-delta) < delta)){
             s = (startx + stopx) / 2;
             mflag=1;
         }
         else{
             mflag=0;
         }

         double fs=gammaCDF(gamma_k, gamma_beta, s) - offset;

         de=root;
         root=stopx;
         froot=fstopx;

         if ((fstartx * fs) < 0){
             stopx=s;
             fstopx=fs;
         }
         else{
             startx = s;
             fstartx = fs;
         }

         if (fabs(fstartx) < fabs(fstopx)){
             //swap startx and stopx
             double tmp=stopx;
             double ftmp=fstopx;
             stopx=startx;
             fstopx=fstartx;
             startx=tmp;
             fstartx=ftmp;
         }
         delta=fabs(stopx-startx);
    }
    return s;
}

#undef GDRCYL_FPMIN
#undef GDRCYL_EPS
#undef GDRCYL_ITMAX

#endif // DMRICM_GDRCYLINDERS_CL
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <limits>
#include <omp.h>
#include <math.h>
#include <cstdlib>

#include "magma_v2.h"
#include "magma_lapack.h"

using namespace std;

void PrimalDualInteriorPoint(float* h_A,float* h_b,float* h_c,magma_int_t m, magma_int_t n,magma_queue_t queue);

int main(int argc, char *argv[]){
    magma_init(); // initialize Magma

    magma_queue_t queue = NULL;
    magma_int_t dev = 0;
    magma_queue_create(dev, &queue);
    magma_int_t m = atoi(argv[1]), n = atoi(argv[2]),lddm = magma_roundup(atoi(argv[1]), 32),lddn = magma_roundup(atoi(argv[2]), 32);
    magma_int_t sizeA = lddm*n, sizeB = m, sizeC = n, sizeSol = n;
    float *h_A, *h_b, *h_c,*h_rand_b,*h_sol;
    magma_smalloc_pinned(&h_A, lddm*n); // host memory for A
    magma_smalloc_pinned(&h_b, lddm); // host memory for B
    magma_smalloc_pinned(&h_rand_b, lddm); // host memory for B
    magma_smalloc_pinned(&h_c, lddn); // host memory for C
    magma_smalloc_pinned(&h_sol, lddn); // host memory for C

    magmaFloat_ptr d_A, d_sol, d_b;
    magma_smalloc(&d_A, lddm*n);
    magma_smalloc(&d_sol, lddn);
    magma_smalloc(&d_b, lddm);
    for(int i = 0; i < 100; i++){


        //1:  uniform (0,1)
        //2:  uniform (-1,1)
        //3:  normal (0,1)
        magma_int_t ione     = 1;
        magma_int_t ISEED[4] = {i,0,i+100,i*2};

        lapackf77_slarnv( &ione, ISEED, &sizeA, h_A );
        lapackf77_slarnv( &ione, ISEED, &sizeB,h_rand_b );
        lapackf77_slarnv( &ione, ISEED, &sizeSol,h_sol );
        lapackf77_slarnv( &ione, ISEED, &sizeC, h_c );

        magma_ssetvector(n, h_sol, 1, d_sol, 1, queue); // copy x -> d_x
        magma_ssetvector(m, h_b, 1, d_b, 1, queue); // copy b -> d_b
        magma_ssetmatrix( m, n, h_A, lddm, d_A, lddm, queue);

        magma_sgemv(MagmaNoTrans,m,n,1.0,d_A,lddm,d_sol,1,0.0,d_sol,1,queue);
        magma_sgetvector(m, d_sol, 1, h_b, 1, queue);

        for(int i = 0; i < m; i++)
            h_b[i] = h_b[i] + h_rand_b[i];

        PrimalDualInteriorPoint(h_A,h_b,h_c,m,n,queue);
    }
    magma_free_pinned(h_A);
    magma_free_pinned(h_b);
    magma_free_pinned(h_c);
    magma_free_pinned(h_rand_b);
    magma_free_pinned(h_sol);

    magma_free(d_A);
    magma_free(d_sol);
    magma_free(d_b);

    magma_queue_destroy(queue);
    magma_finalize();
    return 0;
}

void PrimalDualInteriorPoint(float* h_A,float* h_b,float* h_c,magma_int_t m, magma_int_t n,magma_queue_t queue){
    real_Double_t dev_time,iteration_time;

    float unbounded_break = 1e4,infeasible_break = 1e10*1.0,norm_B,norm_C,gamma = 1e10*1.0, epsilon = .01,mu,current_gamma = gamma;
    magma_int_t theta_dim = (2*n+m);
    dev_time = magma_sync_wtime( queue );
    magma_int_t lddm = magma_roundup(m, 32),lddn = magma_roundup(n, 32), ldd_theta = magma_roundup(2*n+m,32),counter = 0;
    bool decomp_flag = true;
    float *h_x,*h_p,*h_s,*h_theta,
            *h_pr,*h_dr,*h_allR, *h_deltas,*h_phi;
    magma_smalloc_pinned(&h_x,lddn); // host memory for A
    magma_smalloc_pinned(&h_p, lddm); // host memory for B
    magma_smalloc_pinned(&h_s, lddn); // host memory for C
    magma_smalloc_pinned(&h_phi, lddn); // host memory for C
    magma_smalloc_pinned(&h_pr, lddm);
    magma_smalloc_pinned(&h_dr, lddn);
    magma_smalloc_pinned(&h_allR, ldd_theta);
    magma_smalloc_pinned(&h_deltas, ldd_theta);
    magma_smalloc_pinned(&h_theta,ldd_theta*theta_dim);

    #pragma omp parallel for
    for(int i = 0; i < n; i++)
        h_x[i] = h_s[i] = 1.0;
    #pragma omp parallel for
    for(int i = 0; i < m; i++)
        h_p[i] = 1.0;
    #pragma omp parallel for
    for(int i = 0; i < theta_dim*theta_dim; i++)
        h_theta[i] = 0.0;

    magmaFloat_ptr d_s, d_x, d_p,d_b,d_c,d_A,
             d_allR, d_theta, d_deltas;
    magma_smalloc(&d_s, lddn); // device memory for s
    magma_smalloc(&d_x, lddn); // device memory for x
    magma_smalloc(&d_p, lddm); // device memory for p
    magma_smalloc(&d_b, lddm); // device memory for b
    magma_smalloc(&d_c, lddn); // device memory for c
    magma_smalloc(&d_A, lddm*n); // device memory for A
    magma_smalloc(&d_theta, ldd_theta*theta_dim); // device memory for A
    magma_smalloc(&d_allR, ldd_theta);
    magma_smalloc(&d_deltas, ldd_theta);

    magma_ssetvector(n, h_s, 1, d_s, 1, queue); // copy s -> d_s
    magma_ssetvector(n, h_x, 1, d_x, 1, queue); // copy x -> d_x
    magma_ssetvector(m, h_p, 1, d_p, 1, queue); // copy p -> d_p
    magma_ssetvector(n, h_c, 1, d_c, 1, queue); // copy c -> d_c
    magma_ssetvector(m, h_b, 1, d_b, 1, queue); // copy b -> d_b
    magma_ssetmatrix( m, n, h_A, lddm, d_A, lddm, queue);

    //adds in [A 0 0]
    #pragma omp parallel for
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            h_theta[j*ldd_theta + i] = h_A[j*lddm+i];
        }
    }
    //adds in [0 A' I]
    #pragma omp parallel for
    for(int j = n; j < m+n; j++){
        for(int i = m; i < m+n; i++){
            h_theta[j*ldd_theta+i] = h_A[(j-n)+lddm*(i-m)];
        }
    }
    #pragma omp parallel for
    for(int i = 0; i < n; i++)
        h_theta[ (n+m+i)*ldd_theta+i+m] = 1.0;

    norm_C = magma_snrm2(n,d_c,1,queue);
    norm_B = magma_snrm2(m,d_b,1,queue);

    while(gamma>epsilon){
        iteration_time = magma_sync_wtime( queue );
        counter++;
        current_gamma = gamma;
        gamma = magma_sdot(n, d_s, 1, d_x, 1, queue);
        mu = .3*gamma/n;

        //adds in the [S 0 X]
        #pragma omp parallel for
        for(int i = 0; i < n; i++){
            h_theta[ i*ldd_theta+i+n+m] = h_s[i];
            h_theta[ (n+m)*ldd_theta + i*ldd_theta+i+m+n] = h_x[i];
            //for random residuals
            h_allR[m+n+i] = mu-h_x[i]*h_s[i];
            //set up for dual residuals
            h_phi[i] = h_c[i] - h_s[i];
        }

        // for primal residuals
        magma_sgemv(MagmaNoTrans, m, n, -1.0, d_A, lddm, d_x, 1, 1.0, d_b, 1, queue);
        magma_sgetvector(m, d_b, 1, h_pr, 1, queue);
        //for dual residuals
        magma_ssetvector(n, h_phi, 1, d_s, 1, queue); // copy s -> d_s
        magma_sgemv(MagmaTrans, m, n, -1.0, d_A, lddm, d_p, 1, 1.0, d_s, 1, queue);
        magma_sgetvector(n, d_s, 1, h_dr, 1, queue);

        //infeasibility check
        if (magma_snrm2(m,d_b,1,queue) >  infeasible_break* norm_B ||
        magma_snrm2(n,d_s,1,queue) > infeasible_break* norm_C ){
//            cout<<endl<<"Potential infeasible, terminating program. If you are certain this is feasible, increase infeasible_break"<<endl;
            magma_free(d_x);
            magma_free(d_p);
            magma_free(d_s);
            magma_free(d_b);
            magma_free(d_c);
            magma_free(d_A);
            magma_free(d_allR);
            magma_free(d_theta);
            magma_free(d_deltas);

            magma_free_pinned(h_x);
            magma_free_pinned(h_p);
            magma_free_pinned(h_s);
            magma_free_pinned(h_pr);
            magma_free_pinned(h_dr);
            magma_free_pinned(h_allR);
            magma_free_pinned(h_theta);
            magma_free_pinned(h_deltas);
            magma_free_pinned(h_phi);
            return;
        }
        #pragma omp parallel for
        for(int i = 0; i < m; i++)
            h_allR[i] = h_pr[i];
        #pragma omp parallel for
        for(int i = 0; i < n; i++)
            h_allR[i+m] = h_dr[i];


        magma_ssetvector(m, h_b, 1, d_b, 1, queue); // copy b -> d_b
        magma_ssetvector(theta_dim, h_allR, 1, d_allR, 1, queue); // copy b -> d_b
        magma_ssetmatrix(theta_dim,theta_dim, h_theta, ldd_theta, d_theta, ldd_theta, queue);

        magma_int_t *ipiv=NULL,info;
        magma_imalloc_pinned( &ipiv, ldd_theta );
        //not doing anything
        
        magma_sgesv_gpu(theta_dim, 1, d_theta, ldd_theta, ipiv, d_allR, ldd_theta, &info);

        magma_free_pinned(ipiv);
        if(info != 0){
//            cout<<"error with solving"<<endl;
            magma_free(d_x);
            magma_free(d_p);
            magma_free(d_s);
            magma_free(d_b);
            magma_free(d_c);
            magma_free(d_A);
            magma_free(d_allR);
            magma_free(d_theta);
            magma_free(d_deltas);

            magma_free_pinned(h_x);
            magma_free_pinned(h_p);
            magma_free_pinned(h_s);
            magma_free_pinned(h_pr);
            magma_free_pinned(h_dr);
            magma_free_pinned(h_allR);
            magma_free_pinned(h_theta);
            magma_free_pinned(h_deltas);
            magma_free_pinned(h_phi);
            return;
        }
        magma_sgetvector(theta_dim, d_allR, 1, h_deltas, 1, queue);

        float beta_primal = 1.0,beta_dual = 1.0,beta_temp;

        #pragma omp parallel for
        for(int i = 0; i < n; i ++){
            #pragma omp critical
            if(h_deltas[i]<0){
                beta_temp = -h_x[i]/h_deltas[i];

                if(beta_temp < beta_primal)
                    beta_primal = beta_temp;
            }
            if(h_deltas[m+n+i]<0){
                beta_temp = -h_s[i]/h_deltas[m+n+i];
                if(beta_temp < beta_dual)
                    beta_dual = beta_temp;
            }
        }

        //looks dumb but needed for openMP multithreading
        bool end_early = false;
        #pragma omp parallel for
        for(int i = 0; i < n; i ++){
            h_x[i] = h_x[i] + beta_primal*h_deltas[i];
            h_s[i] = h_s[i] + beta_dual*h_deltas[n+m+i];
            #pragma omp critical
            if (h_x[i] > unbounded_break){
//                cout<<endl<<"Potentially unbounded linear program, increase unbounded_break if you're certain the LP is bounded"<<endl;
                end_early = true;
            }
        }
        if(counter == 40){
//	        cout<<"failed to converge"<<endl;
            magma_free(d_x);
            magma_free(d_p);
            magma_free(d_s);
            magma_free(d_b);
            magma_free(d_c);
            magma_free(d_A);
            magma_free(d_allR);
            magma_free(d_theta);
            magma_free(d_deltas);

            magma_free_pinned(h_x);
            magma_free_pinned(h_p);
            magma_free_pinned(h_s);
            magma_free_pinned(h_pr);
            magma_free_pinned(h_dr);
            magma_free_pinned(h_allR);
            magma_free_pinned(h_theta);
            magma_free_pinned(h_deltas);
            magma_free_pinned(h_phi);
            return;}

        #pragma omp parallel for
        for(int i = 0; i < m; i ++)
            h_p[i] = h_p[i] + beta_dual*h_deltas[n+i];

        magma_ssetvector(n, h_s, 1, d_s, 1, queue); // copy s -> d_s
        magma_ssetvector(m, h_p, 1, d_p, 1, queue); // copy p -> d_p
        magma_ssetvector(n, h_x, 1, d_x, 1, queue); // copy x -> d_x
        iteration_time =magma_sync_wtime( queue ) - iteration_time;
    }
    magma_sgetvector(m, d_x, 1, h_x, 1, queue);
    cout <<counter<< ','<<magma_sync_wtime( queue ) - dev_time<<','<<iteration_time<<endl;
//    for( int i = 0; i < n; i++)
//        cout<<h_x[i]<<endl;

    // Clean up
    magma_free(d_x);
    magma_free(d_p);
    magma_free(d_s);
    magma_free(d_b);
    magma_free(d_c);
    magma_free(d_A);
    magma_free(d_allR);
    magma_free(d_theta);
    magma_free(d_deltas);
    
    magma_free_pinned(h_x);
    magma_free_pinned(h_p);
    magma_free_pinned(h_s);
    magma_free_pinned(h_pr);
    magma_free_pinned(h_dr);
    magma_free_pinned(h_allR);
    magma_free_pinned(h_theta);
    magma_free_pinned(h_deltas);
    magma_free_pinned(h_phi);
}

#include <cblas.h>
#include <stdio.h>
#include <lapacke.h>
#include <omp.h>
#include <chrono>
#include<iostream>
#include <cstdlib>

using namespace std;

void PrimalDualInteriorPoint(float* A,float* b,float* c,int m, int n);
void print_matrix(float* A, int m, int n);

int main(int argc, char *argv[]){
    int m = atoi(argv[1]), n = atoi(argv[2]), ione = 1;
    float *A = new float[m*n], *b = new float[m], *c = new float[n], *rand_pertebation = new float[m], *sol = new float[n];

    for(int i = 0 ; i < 100; i++){
        int ISEED[4] = {i,0,i+100,i*2}, sizeA = m*n, sizec = n, sizeSol = n, sizeRand = m;

        LAPACKE_slarnv ( ione, ISEED, sizeA, A );
        LAPACKE_slarnv ( ione, ISEED, sizec,c );
        LAPACKE_slarnv ( ione, ISEED, sizeSol, sol );
        LAPACKE_slarnv ( ione, ISEED, sizeRand, rand_pertebation );

        cblas_sgemv(CblasColMajor, CblasNoTrans, m, n, 1.0, A, m, sol, 1, 0.0, b,1);

        for(int i = 0; i < m; i++)
            b[i] = b[i] + rand_pertebation[i];

        PrimalDualInteriorPoint(A,b,c,m,n);
    }
    delete []A;
    delete []b;
    delete []c;
    delete []rand_pertebation;
    delete []sol;
    return 0;
}

void PrimalDualInteriorPoint(float* A,float* b,float* c,int m, int n){
    float unbounded_break = 1e4,infeasible_break = 1e10*1.0,norm_B,norm_C,gamma = 1e10*1.0, epsilon = .01,mu,current_gamma = gamma;
    int theta_dim = (2*n+m),counter = 0,info;

    float *x = new float[n], *p = new float[m], *s = new float[n], *theta = new float[theta_dim*theta_dim],
        *temp_theta = new float[theta_dim*theta_dim ], *pr = new float[m], *dr = new float[n], *allR = new float[theta_dim];
    int *ipiv = new int[theta_dim];

    #pragma omp parallel for
    for(int i = 0; i < n; i++)
        x[i] = s[i] = 1.0;
    #pragma omp parallel for
    for(int i = 0; i < m; i++)
        p[i] = 1.0;
    #pragma omp parallel for
    for(int i = 0; i < theta_dim*theta_dim; i++)
        theta[i] = 0.0;

    //adds in [A 0 0]
    #pragma omp parallel for
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            theta[j*theta_dim + i] = A[j*m+i];
        }
    }
    //adds in [0 A' I]
    #pragma omp parallel for
    for(int j = n; j < m+n; j++){
        for(int i = m; i < m+n; i++){
            theta[j*theta_dim+i] = A[(j-n)+m*(i-m)];
        }
    }
    #pragma omp parallel for
    for(int i = 0; i < n; i++)
        theta[ (n+m+i)*theta_dim+i+m] = 1.0;

    norm_C = cblas_snrm2 (n,c,1);
    norm_B = cblas_snrm2 (n,b,1);
    auto super_start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::high_resolution_clock::now();
    while(gamma>epsilon){
        // screw the chrono package, im not gonna spend time trying to figuring out what this means
        start = std::chrono::high_resolution_clock::now();
        counter++;
        current_gamma = gamma;
        gamma = cblas_sdot(n, s, 1, x, 1);
        mu = .3*gamma/n;

        //adds in the [S 0 X]
        #pragma omp parallel for
        for(int i = 0; i < n; i++){
            theta[ i*theta_dim+i+n+m] = s[i];
            theta[ (n+m)*theta_dim + i*theta_dim+i+m+n] = x[i];
            //for random residuals
            allR[m+n+i] = mu-x[i]*s[i];
            dr[i] = c[i] - s[i];
        }

        #pragma omp parallel for
        for(int i = 0; i < m; i++)
            pr[i] = b[i];

        // for primal residuals, before sgemv pr = b so it works out
        cblas_sgemv(CblasColMajor, CblasNoTrans, m, n, -1.0, A, m, x, 1, 1.0, pr,1);
        #pragma omp parallel for
        for(int i = 0; i < m; i++)
            allR[i] = pr[i];
        //for dual residuals, before sgemv dr = c - s so it works out
        cblas_sgemv(CblasColMajor, CblasTrans, m, n, -1.0, A, m, p, 1, 1.0, dr,1);
        #pragma omp parallel for
        for(int i = 0; i < n; i++)
            allR[m+i] = dr[i];

        //infeasibility check
        if (cblas_snrm2 (m,pr,1) >  infeasible_break*norm_B ||
        cblas_snrm2 (n,dr,1) > infeasible_break*norm_C ){
//            cout<<endl<<"Potential infeasible, terminating program. If you are certain this is feasible, increase infeasible_break"<<endl;
            delete []x;
            delete []p;
            delete []s;
            delete []theta;
            delete []temp_theta;
            delete []pr;
            delete []dr;
            delete []allR;
            delete []ipiv;
            return;
        }

        //allR now becomes deltas
        info = LAPACKE_slacpy(LAPACK_COL_MAJOR,'_',theta_dim,theta_dim,theta,theta_dim,temp_theta,theta_dim);
        info = LAPACKE_sgesv (LAPACK_COL_MAJOR,theta_dim,1,temp_theta,theta_dim,ipiv,allR,theta_dim);
        if(info != 0){
//            cout<<"error with solving "<<info<<endl;
            delete []x;
            delete []p;
            delete []s;
            delete []theta;
            delete []temp_theta;
            delete []pr;
            delete []dr;
            delete []allR;
            delete []ipiv;
            return;
        }

        float beta_primal = 1.0,beta_dual = 1.0,beta_temp;

        #pragma omp parallel for
        for(int i = 0; i < n; i ++){
            #pragma omp critical
            if(allR[i]<0){
                beta_temp = -x[i]/allR[i];

                if(beta_temp < beta_primal)
                    beta_primal = beta_temp;
            }
            if(allR[m+n+i]<0){
                beta_temp = -s[i]/allR[m+n+i];
                if(beta_temp < beta_dual)
                    beta_dual = beta_temp;
            }
        }

        #pragma omp parallel for
        for(int i = 0; i < n; i ++){
            x[i] = x[i] + beta_primal*allR[i];
            s[i] = s[i] + beta_dual*allR[n+m+i];
        }

        #pragma omp parallel for
        for(int i = 0; i < m; i ++)
            p[i] = p[i] + beta_dual*allR[n+i];
        end = std::chrono::high_resolution_clock::now();
        if (counter > 50){
            delete []x;
            delete []p;
            delete []s;
            delete []theta;
            delete []temp_theta;
            delete []pr;
            delete []dr;
            delete []allR;
            delete []ipiv;
            return;
        }
    }
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto super_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - super_start);
    cout <<counter<< ','<<super_elapsed.count()<<','<<elapsed.count()<<endl;
    // Clean up
    delete []x;
    delete []p;
    delete []s;
    delete []theta;
    delete []temp_theta;
    delete []pr;
    delete []dr;
    delete []allR;
    delete []ipiv;
}

void print_matrix(float* A, int m, int n) {
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            cout<<A[m*j+i]<<" ";
        }
        cout<<endl;
    }
}

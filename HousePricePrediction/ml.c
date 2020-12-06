/* 
 * 
 * This code calculates the house price of a house by learing from
 * training data. It uses pseudo inverse of a given matrix to find the 
 * weight of different features.
 * 
 * Predicted Price : Y = W0 + W1*x1 + W2*X2 + W3*X3 + W4*X4
 * Weight Matrix : W = pseudoInv(X)*Y
 * pseudoInv(X) = inverse(transpose(X)*X) * transpose(X)  
 * 
 * weight(w) = pseudoInv(X) * Y
 * 			where	X = Input data matrix
 * 					Y = Target vector
 * 
 */
 
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

// all methods declarations
double** multiplyMatrix(double **matA, double **matB, int r1, int c1, int r2, int c2); //function for multiplying 2 matrices
double** transposeMatrix(double** mat, int row, int col); //function to calculate the transpose of a matrix
double** inverseMatrix(double **matA, int dimension); //function to calculate the inverse of a matrix
void free2D(double** arr, int numRows); //function to free heap allocations of a 2D array

// main method starts here
int main(int argc, char** argv){

    FILE* trainFile = fopen(argv[1], "r"); //training file to train data and use it to predict the price of houses from the given test data file
    FILE* testFile = fopen(argv[2], "r"); //test file to predict the price of the house based on the data provided from training file
    if(trainFile == NULL || testFile == NULL){
        fclose(trainFile);
        fclose(testFile);
        return 0;
    }
    int trainRows = -1;
    int trainCols = -1;
    fscanf(trainFile, "%d\n%d", &trainCols, &trainRows);
    
    
    
    double** X = malloc(trainRows*sizeof(double*));
    for(int i = 0; i < trainRows; i++){
        X[i] = malloc((trainCols+1)*sizeof(double));
    }
    for(int i = 0; i < trainRows; i++){
        for(int j = 0; j < (trainCols+1); j++){
            X[i][j] = 0;
        }
    }
    for(int i = 0; i < trainRows; i++){
        X[i][0] = 1;
    }
    
    
    
    double** Y = malloc(trainRows*sizeof(double*));
    for(int i = 0; i < trainRows; i++){
        Y[i] = malloc(1*sizeof(double));
    }
    for(int i = 0; i < trainRows; i++){
        Y[i][0] = 0;
    }
    
    
    
    
        int sInd = 0;
        int Xr = 0;
        int Yr = 0;
        while(1){
        
            char curr[700];
            int n = fscanf(trainFile, "%s", curr);
            int Xc = 1;
            if(n != 1){
                break;
            }
            //converting the string input to double values and storing in the X variable
            for(int i = 0; i < strlen(curr); i++){
                if(curr[i] == ','){
                    int length = i - sInd;
                    char* temp = malloc(length*sizeof(char));
                    for(int k = sInd; k < i; k++){
                        temp[k-sInd] = curr[k];
                    }
                    double xVal = strtod(temp, NULL);
                    X[Xr][Xc] = xVal;
                    Xc++;
                    sInd = i+1;
                    free(temp);
                }
            }
            
            
            //last element to be stored from the string input
            char *te = malloc((strlen(curr) - sInd)*sizeof(char));
            for(int i = sInd; i < strlen(curr); i++){
                te[i-sInd] = curr[i]; 
            }
            double yVal = strtod(te, NULL);
            Y[Yr][0] = yVal;
            Yr++;
            Xr++;
            sInd = 0;
            free(te);
            
        }
        
        
        double** Xtran = transposeMatrix(X, trainRows, (trainCols+1)); //X Transpose
        double** XtranbyX = multiplyMatrix(Xtran, X, (trainCols+1), trainRows, trainRows, (trainCols+1)); //Multiplying X transpose by X
        double** pseudoX = inverseMatrix(XtranbyX, (trainCols+1)); //taking the inverse of X transpose by X
        double** pseudoXbyXtran = multiplyMatrix(pseudoX, Xtran, (trainCols+1), (trainCols+1), (trainCols+1), trainRows); //multiplying pseudoX by X transpose
        double** W = multiplyMatrix(pseudoXbyXtran, Y, (trainCols+1), trainRows, trainRows, 1); // multiplying everything by Y and storing it into the varible W
        fclose(trainFile);
        
        
        
        //predicting the price from the given test data
        double price = W[0][0];
        int numHouses;
        fscanf(testFile, "%d", &numHouses);
        for(int i = 0; i < numHouses; i++){
            char curr[700];
            int hsInd = 0;
            int hInd = 0;
            fscanf(testFile, "%s", curr);
            //array to store the values in the testfiles
            double* houseVals = malloc(trainCols*sizeof(double));
            for(int i = 0; i < trainCols; i++){
                houseVals[i] = 0;
            }
            //converting the string input from the test files into double values and storing it in the houseVals variable
            for(int i = 0; i < strlen(curr); i++){
                if(curr[i] == ','){
                    int length = i - hsInd;
                    char* temp = malloc(length*sizeof(char));
                    for(int k = hsInd; k < i; k++){
                        temp[k-hsInd] = curr[k];
                    }
                    double hVal = strtod(temp, NULL);
                    houseVals[hInd] = hVal;
                    hInd++;
                    hsInd = i + 1;
                    free(temp);
                }
            }
            //last element to be stored in the houseVals variable
            char*te = malloc((strlen(curr)-hsInd)*sizeof(char));
            for(int i = hsInd; i < strlen(curr); i++){
                te[i-hsInd] = curr[i];
            }
            double lastHVal = strtod(te, NULL);
            houseVals[hInd] = lastHVal;
            free(te);
            //evaluating and printing the price of the house from the test data provided
            for(int i = 1; i < trainCols+1; i++){
                price = price + (houseVals[i-1]*W[i][0]);
            }
            printf("%0.0lf\n", price);
            price = W[0][0];
            free(houseVals);
        }
        
        
        fclose(testFile);
        //freeing all the heap allocations
        free2D(W, (trainCols+1));
        free2D(pseudoXbyXtran, (trainCols+1));
        free2D(pseudoX, (trainCols+1));
        free2D(XtranbyX, (trainCols+1));
        free2D(Xtran, (trainCols+1));
        free2D(Y, trainRows);
        free2D(X, trainRows);
    return 0;
	
}

double** multiplyMatrix(double **matA, double **matB, int r1, int c1, int r2, int c2)
{
    if(c1 != r2){
        return NULL;
    }
    
    double** result=malloc(r1*sizeof(double*)); 
    
    // your code goes here 
    for(int i = 0; i < r1; i++){
        result[i] = malloc(c2*sizeof(double));
    }
    for(int i = 0; i < r1; i++){
        for(int j = 0; j < c2; j++){
            result[i][j] = 0;
        }
    }
    //setting each element to zero
    double s = 0.000000;
    //evaluating the multiplication function
    for(int i = 0; i < r1; i++){
        for(int j = 0; j < c2; j++){
            for(int k = 0; k < r2; k++){
                s += matA[i][k] * matB[k][j];
            }
            result[i][j] = s;
            s = 0.000000;
        }
    }
    return result;
}


double** transposeMatrix(double** mat, int row, int col)
{
  
	double** matTran=malloc(col*sizeof(double*)); 
    
    // your code goes here
    for(int i = 0; i < col; i++){
        matTran[i] = malloc(row*sizeof(double));
    }
    for(int i = 0; i < col; i++){
        for(int j = 0; j < row; j++){
            matTran[i][j] = 0;
        }
    }
    for(int i = 0; i < col; i++){
        for(int j = 0; j < row; j++){
            matTran[i][j] = mat[j][i];
        }
    }
    return matTran;        
}


double** inverseMatrix(double **matA, int dimension)
{

    double** matI=malloc(dimension*sizeof(double*)); 
    // your code goes here
    for(int i = 0; i < dimension; i++){
        matI[i] = malloc(dimension*sizeof(double));
    }
    
    for(int i = 0; i < dimension; i++){
        for(int j = 0; j < dimension; j++){
            if( i == j){
                matI[i][j] = 1.000000;
            }else{
                matI[i][j] = 0.000000;
            }
        }
    }
    
    for(int p = 0; p < dimension; p++){
        //variable f storing the diagnol values
        double f = matA[p][p];
        //function to divide row p of matA and matI by f
        for(int c = 0; c < dimension; c++){
            matA[p][c] = (matA[p][c])/f;
            matI[p][c] = (matI[p][c])/f;
        }
        for(int i = (p+1); i < dimension; i++){
            f = matA[i][p];
            //function for subtracting row matA[p] * f from row matA[i] 
            //also a function for subtracting row matI[p] * f from row matI[i]
            for(int subA = 0; subA < dimension; subA++){
                matA[i][subA] = matA[i][subA] - (f * matA[p][subA]);
                matI[i][subA] = matI[i][subA] - (f * matI[p][subA]);
            }
        }
    }
    for(int p = (dimension - 1); p >= 0; p--){
        for(int i = (p - 1); i >= 0; i--){
            double f = matA[i][p];
            //function for subtracting row matA[p] * f from row matA[i] 
            //also a function for subtracting row matI[p] * f from row matI[i]
            for(int subA = 0; subA < dimension; subA++){
                matA[i][subA] = matA[i][subA] - (f * matA[p][subA]);
                matI[i][subA] = matI[i][subA] - (f * matI[p][subA]);
            }
        }
    }
	return matI;
}
void free2D(double** arr, int numRows){
    //freeing each 1d array inside the 2d array
    for(int i = 0; i < numRows; i++){
        free(arr[i]);
    }
    //freeing the 2d array itself
    free(arr);
}
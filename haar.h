#include <cstdlib>
#include <cmath>

void haar1(float *vec, int n, int w)
{
int i=0;
float *vecp = (float *)malloc(sizeof(float)*n);
for(i=0;i<n;i++)
    vecp[i] = 0;

    w/=2;
    for(i=0;i<w;i++)
    {
        vecp[i] = (vec[2*i] + vec[2*i+1])/sqrt(2.0);
        vecp[i+w] = (vec[2*i] - vec[2*i+1])/sqrt(2.0);
    }

    for(i=0;i<(w*2);i++)
        vec[i] = vecp[i];

//      delete [] vecp;
}
/** The 2D Haar Transform **/
void haar2(float **matrix, int rows, int cols)
{
float *temp_row = (float *)malloc(sizeof(float)*cols);
float *temp_col = (float *)malloc(sizeof(float)*rows);

int i=0,j=0;
int w = cols, h=rows;
while(w>1 || h>1)
{
    if(w>1)
    {
        for(i=0;i<h;i++)
        {
            for(j=0;j<cols;j++)
                temp_row[j] = matrix[i][j];

            haar1(temp_row,cols,w);

            for(j=0;j<cols;j++)
                matrix[i][j] = temp_row[j];
        }
    }

    if(h>1)
    {
        for(i=0;i<w;i++)
        {
            for(j=0;j<rows;j++)
                temp_col[j] = matrix[j][i];
            haar1(temp_col, rows, h);
            for(j=0;j<rows;j++)
                matrix[j][i] = temp_col[j];
        }
    }

    if(w>1)
        w/=2;
    if(h>1)
        h/=2;
}
}
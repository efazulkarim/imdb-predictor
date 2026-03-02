#include<bits/stdc++.h>

using namespace std;

#define lol long long int
#define MOD 1000000007
#define MAX 1005

lol dp[MAX+7][MAX+7];

void build_dp()
{
    dp[0][0] = 1;


    for(int i=0; i<MAX; i++)
        for(int j=0; j<MAX; j++)
        {
            dp[i][j+2] += dp[i][j];
            dp[i+1][j+1] += dp[i][j];
            dp[i+2][j+2] += dp[i][j];
            dp[i+1][j] += dp[i][j];
            dp[i][j+2] %= MOD;
            dp[i+1][j+1] %= MOD;
            dp[i+2][j+2] %= MOD;
            dp[i+1][j] %= MOD;
        }
}
int main()
{
    build_dp();
    srand(time(0));
    int x,y,N = 20000;
    FILE *in = fopen("1.in","w");
    FILE *out = fopen("1.out","w");
    while(N--)
    {
        x = rand()%1000+1;
        y = rand()%1000+1;
        fprintf(in,"%d %d\n",x,y);
        fprintf(out,"%lld\n",dp[x][y]);
    }
    //fprintf(in,"%d %d",0,0);
    fclose(in);
    fclose(out);
    return 0;
}

// Heuristic handwritten digit recognizer for 28x28 grayscale images
// The OJ will include this header and call judge(IMAGE_T&).
// IMAGE_T is defined in read_support.hpp as std::vector<std::vector<double>> with values in [0,1], 1=white, 0=black.

#ifndef JUDGE_H
#define JUDGE_H
#include "read_support.hpp"
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <utility>

namespace detail_digit {
static int H = 28, W = 28;

static inline std::vector<std::vector<int> > binarize(const IMAGE_T &img) {
    std::vector<std::vector<int> > b(H, std::vector<int>(W, 0));
    // Otsu-like simple threshold: use mean as base and clamp to reasonable range
    double sum = 0.0; int cnt = 0;
    for (int i = 0; i < H; ++i) for (int j = 0; j < W; ++j) { sum += img[i][j]; ++cnt; }
    double mean = cnt ? (sum / cnt) : 0.5;
    double th = 1.0 - mean * 0.9; // expect digit darker
    if (th < 0.2) th = 0.2; else if (th > 0.7) th = 0.7;
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j)
            b[i][j] = (img[i][j] < th) ? 1 : 0; // 1 means foreground (dark stroke)
    return b;
}

static inline void bbox(const std::vector<std::vector<int> > &b, int &r0,int &c0,int &r1,int &c1){
    r0=H; c0=W; r1=-1; c1=-1;
    for(int i=0;i<H;++i) for(int j=0;j<W;++j) if(b[i][j]){ r0=std::min(r0,i); c0=std::min(c0,j); r1=std::max(r1,i); c1=std::max(c1,j);}    
    if(r1<r0){ r0=c0=0; r1=H-1; c1=W-1; }
}

static inline int count_holes(const std::vector<std::vector<int> > &b, int r0,int c0,int r1,int c1){
    // Flood fill background and count enclosed zero regions in inverted mask
    int h=r1-r0+1, w=c1-c0+1;
    std::vector<std::vector<int> > vis(h, std::vector<int>(w,0));
    // Mark background connected to border in complement graph
    std::queue<std::pair<int,int> > q;
    for(int i=0;i<h;++i){
        if(b[r0+i][c0+0]==0){ q.push(std::make_pair(i,0)); vis[i][0]=1; }
        if(b[r0+i][c0+w-1]==0){ q.push(std::make_pair(i,w-1)); vis[i][w-1]=1; }
    }
    for(int j=0;j<w;++j){
        if(b[r0+0][c0+j]==0){ q.push(std::make_pair(0,j)); vis[0][j]=1; }
        if(b[r0+h-1][c0+j]==0){ q.push(std::make_pair(h-1,j)); vis[h-1][j]=1; }
    }
    int dr[4]={1,-1,0,0}, dc[4]={0,0,1,-1};
    while(!q.empty()){
        std::pair<int,int> pc = q.front(); q.pop();
        int r = pc.first, c = pc.second;
        for(int k=0;k<4;++k){
            int nr=r+dr[k], nc=c+dc[k];
            if(nr>=0 && nr<h && nc>=0 && nc<w && !vis[nr][nc] && b[r0+nr][c0+nc]==0){
                vis[nr][nc]=1; q.push(std::make_pair(nr,nc));
            }
        }
    }
    // Remaining background components are holes
    int holes=0;
    for(int i=0;i<h;++i) for(int j=0;j<w;++j){
        if(!vis[i][j] && b[r0+i][c0+j]==0){
            ++holes; vis[i][j]=1; q.push(std::make_pair(i,j));
            while(!q.empty()){
                std::pair<int,int> pc = q.front(); q.pop();
                int r = pc.first, c = pc.second;
                for(int k=0;k<4;++k){
                    int nr=r+dr[k], nc=c+dc[k];
                    if(nr>=0 && nr<h && nc>=0 && nc<w && !vis[nr][nc] && b[r0+nr][c0+nc]==0){
                        vis[nr][nc]=1; q.push(std::make_pair(nr,nc));
                    }
                }
            }
        }
    }
    return holes;
}

static inline int vertical_cuts(const std::vector<std::vector<int> > &b,int r0,int c0,int r1,int c1){
    int cuts=0; bool in=false;
    for(int j=c0;j<=c1;++j){ int col=0; for(int i=r0;i<=r1;++i) col+=b[i][j]; if(col==0){ if(!in){in=true; ++cuts;} } else in=false; }
    return cuts; // number of blank vertical gaps inside bbox
}

static inline int endpoints(const std::vector<std::vector<int> > &b,int r0,int c0,int r1,int c1){
    int ep=0; int dr[8]={1,1,1,0,0,-1,-1,-1}; int dc[8]={-1,0,1,-1,1,-1,0,1};
    for(int i=r0;i<=r1;++i) for(int j=c0;j<=c1;++j) if(b[i][j]){
        int deg=0; for(int k=0;k<8;++k){ int ni=i+dr[k], nj=j+dc[k]; if(ni>=r0&&ni<=r1&&nj>=c0&&nj<=c1&&b[ni][nj]) ++deg; }
        if(deg<=1) ++ep;
    }
    return ep;
}

static inline double fill_ratio(const std::vector<std::vector<int> > &b,int r0,int c0,int r1,int c1){
    int area=(r1-r0+1)*(c1-c0+1), on=0; for(int i=r0;i<=r1;++i) for(int j=c0;j<=c1;++j) on+=b[i][j]; return area? (double)on/area : 0.0;
}

static inline int classify(const std::vector<std::vector<int> > &b){
    int r0,c0,r1,c1; bbox(b,r0,c0,r1,c1);
    int h=r1-r0+1, w=c1-c0+1; double ar = h>0? (double)w/h : 1.0;
    int holes = count_holes(b,r0,c0,r1,c1);
    int vcuts = vertical_cuts(b,r0,c0,r1,c1);
    int eps = endpoints(b,r0,c0,r1,c1);
    double fill = fill_ratio(b,r0,c0,r1,c1);

    // Rule-based simple classifier tuned for MNIST-ish digits
    if(holes>=2) return 8;
    if(holes==1){
        // Distinguish 0,6,9 based on endpoint count and aspect
        if(eps<=4){
            if(ar>0.9 && ar<1.2) return 0; // near circular
            // check top vs bottom density
            int top=0,bottom=0; for(int i=r0;i<=r0+(h/2);++i) for(int j=c0;j<=c1;++j) top+=b[i][j];
            for(int i=r0+(h/2);i<=r1;++i) for(int j=c0;j<=c1;++j) bottom+=b[i][j];
            if(bottom>top) return 6; else return 9;
        } else {
            // Open loop like 6 or 9
            int left=0,right=0; for(int i=r0;i<=r1;++i) for(int j=c0;j<=c0+w/2;++j) left+=b[i][j]; for(int i=r0;i<=r1;++i) for(int j=c0+w/2;j<=c1;++j) right+=b[i][j];
            if(left>right) return 6; else return 9;
        }
    }

    // No holes: consider 1,2,3,4,5,7
    // 1: tall, thin, small fill, two endpoints
    if(ar<0.6 && h>18) return 1;

    // 4: often has a vertical gap and more endpoints
    if(vcuts>=1 && eps>=4) return 4;

    // 7: flatter with top bar, few endpoints
    if(ar>0.9 && eps<=6) {
        // top heavier than bottom
        int top=0,bottom=0; for(int i=r0;i<=r0+(h/2);++i) for(int j=c0;j<=c1;++j) top+=b[i][j];
        for(int i=r0+(h/2);i<=r1;++i) for(int j=c0;j<=c1;++j) bottom+=b[i][j];
        if(top>bottom) return 7;
    }

    // 2 vs 3 vs 5: use endpoints and vertical gaps and fill
    if(eps>=6){
        // 3 tends to have fewer vertical blanks
        if(vcuts==0) return 3;
        // 5 often bottom heavy
        int top=0,bottom=0; for(int i=r0;i<=r0+(h/2);++i) for(int j=c0;j<=c1;++j) top+=b[i][j];
        for(int i=r0+(h/2);i<=r1;++i) for(int j=c0;j<=c1;++j) bottom+=b[i][j];
        if(bottom>top) return 5; else return 2;
    }

    // fallback based on simple features
    if(fill < 0.17) return 1;
    if(ar >= 0.8 && ar <= 1.2) return 0;
    return 3; // safe bias
}
} // namespace

int judge(IMAGE_T &img){
    // Defensive: ensure size 28x28
    detail_digit::H = (int)img.size();
    detail_digit::W = detail_digit::H? (int)img[0].size() : 28;
    std::vector<std::vector<int> > b = detail_digit::binarize(img);
    int d = detail_digit::classify(b);
    if(d<0 || d>9) d = 0; // clamp safety
    return d;
}

#endif // JUDGE_H

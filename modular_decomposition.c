#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* Graph (CSR) */

typedef struct {
    int n;
    int m2;
    int *off;
    int *adj;
} Graph;

static Graph *graph_build(int n, const int *deg, int **nbrs){
    Graph *G = (Graph*)calloc(1,sizeof(Graph));
    G->n=n;
    long sum=0; for(int i=0;i<n;i++) sum+=deg[i];
    G->m2=(int)sum;
    G->off=(int*)malloc((n+1)*sizeof(int));
    G->adj=(int*)malloc((G->m2)*sizeof(int));
    int acc=0;
    for(int i=0;i<n;i++){
        G->off[i]=acc;
        memcpy(&G->adj[acc], nbrs[i], deg[i]*sizeof(int));
        acc+=deg[i];
    }
    G->off[n]=acc;
    return G;
}
static void graph_free(Graph *G){ if(!G) return; free(G->off); free(G->adj); free(G); }
#define DEG(v) (G->off[(v)+1]-G->off[(v)])


/* Partition refinement (Paige–Tarjan) */

typedef struct Block Block;
struct Block {
    Block *prevB, *nextB;
    int head, tail;      
    int size;
    int id;
    int stamp;  
    int *picked; int pc, pcap; 
};

typedef struct {
    int n;
    Block *headB;
    Block **blk; 
    int *prev, *next;  
    unsigned char *mark;
    int stamp;
} Partition;

static Block *block_new(int id){
    Block *b=(Block*)calloc(1,sizeof(Block));
    b->head=b->tail=-1; b->size=0; b->id=id; b->stamp=0; b->pc=0; b->pcap=0; b->picked=NULL;
    return b;
}
static void block_push_back(Partition *P, Block *b, int v){
    if(b->tail==-1){ b->head=b->tail=v; P->prev[v]=-1; P->next[v]=-1; }
    else { P->next[b->tail]=v; P->prev[v]=b->tail; P->next[v]=-1; b->tail=v; }
    P->blk[v]=b; b->size++;
}
static void block_remove(Partition *P, Block *b, int v){
    int pv=P->prev[v], nv=P->next[v];
    if(pv!=-1) P->next[pv]=nv; else b->head=nv;
    if(nv!=-1) P->prev[nv]=pv; else b->tail=pv;
    b->size--;

}
static void block_insert_before(Partition *P, Block *at, Block *nw){
    nw->prevB = at? at->prevB : NULL;
    nw->nextB = at;
    if(at){
        if(at->prevB) at->prevB->nextB=nw;
        at->prevB=nw;
    }
    if(P->headB==at || P->headB==NULL) P->headB=nw;
}
static void block_insert_after(Partition *P, Block *at, Block *nw){
    nw->nextB = at? at->nextB : NULL;
    nw->prevB = at;
    if(at){
        if(at->nextB) at->nextB->prevB=nw;
        at->nextB=nw;
    }else{
        P->headB=nw;
    }
}
static Partition *partition_new(int n, const int *order){
    Partition *P=(Partition*)calloc(1,sizeof(Partition));
    P->n=n;
    P->blk=(Block**)malloc(n*sizeof(Block*));
    P->prev=(int*)malloc(n*sizeof(int));
    P->next=(int*)malloc(n*sizeof(int));
    P->mark=(unsigned char*)calloc(n,1);
    P->stamp=1;

    Block *b0=block_new(0);
    for(int i=0;i<n;i++){
        int v = order? order[i] : i;
        block_push_back(P, b0, v);
    }
    P->headB=b0;
    return P;
}
static void partition_free(Partition *P){
    if(!P) return;
    Block *b=P->headB;
    while(b){ Block *n=b->nextB; free(b->picked); free(b); b=n; }
    free(P->mark); free(P->next); free(P->prev); free(P->blk); free(P);
}


typedef struct { Block **a; int n, cap; } Touched;
static void touched_init(Touched *T){ T->a=NULL; T->n=0; T->cap=0; }
static void touched_push(Touched *T, Block *b){
    if(T->n==T->cap){ T->cap=T->cap?2*T->cap:16; T->a=(Block**)realloc(T->a,T->cap*sizeof(Block*)); }
    T->a[T->n++]=b;
}
static void touched_reset(Touched *T){ T->n=0; }
static void touched_free(Touched *T){ free(T->a); }


static void block_push_pick(Block *b, int v){
    if(b->pc==b->pcap){ b->pcap = b->pcap? 2*b->pcap : 8; b->picked=(int*)realloc(b->picked, b->pcap*sizeof(int)); }
    b->picked[b->pc++]=v;
}


static Block *partition_split_picked(Partition *P, Block *blk){
    int picked = blk->pc;
    if(picked==0 || picked==blk->size){ blk->pc=0; return NULL; }

    Block *B2 = block_new(blk->id + 1000000);
    for(int i=0;i<blk->pc;i++){
        int v=blk->picked[i];
        if(P->blk[v]!=blk) continue;
        block_remove(P, blk, v);
        block_push_back(P, B2, v);
    }
    blk->pc=0;

    int left_small = (B2->size <= blk->size);
    if(left_small){ block_insert_before(P, blk, B2); return B2; }
    else{ block_insert_after(P, blk, B2); return blk; }
}


typedef struct { int *q; int n, h, t; } Queue;
static Queue *queue_new(int n){ Queue *Q=(Queue*)malloc(sizeof(Queue)); Q->q=(int*)malloc(n*sizeof(int)); Q->n=n; Q->h=Q->t=0; return Q; }
static void queue_free(Queue *Q){ free(Q->q); free(Q); }
static int  queue_empty(Queue *Q){ return Q->h==Q->t; }
static void enqueue(Queue *Q, int v){ Q->q[Q->t++]=v; if(Q->t==Q->n) Q->t=0; }


/* Factorizing permutation */


typedef struct LBBlock {
    int head, tail;
    struct LBBlock *prev, *next;
    int touched;  
    struct LBBlock *newblk;
} LBBlock;

static LBBlock* lb_new_block(void){
    LBBlock* b=(LBBlock*)calloc(1,sizeof(LBBlock));
    b->head=b->tail=-1; 
    b->prev=b->next=NULL;
    b->touched = 0;
    return b;
}
static void lb_remove_block(LBBlock *b){
    if(b->prev) b->prev->next=b->next;
    if(b->next) b->next->prev=b->prev;
}


static void lb_push_back(LBBlock *b, int v, int *next, int *prev, LBBlock **owner){
    if (b->head == -1){
        b->head = b->tail = v;
        prev[v] = -1;
        next[v] = -1;
    } else {
        prev[v] = b->tail;
        next[b->tail] = v;
        next[v] = -1;
        b->tail = v;
    }
    owner[v] = b;
}


static void lb_detach(LBBlock *b, int v, int *next, int *prev, LBBlock **owner){
    int pv=prev[v], nv=next[v];
    if(pv!=-1) next[pv]=nv; else b->head=nv;
    if(nv!=-1) prev[nv]=pv; else b->tail=pv;
    prev[v]=next[v]=-1; owner[v]=NULL;
}


static LBBlock* lb_split_before(LBBlock *B, unsigned char *marked,
                                int *next, int *prev, LBBlock **owner){
    LBBlock *Bp = lb_new_block();
    B->newblk = Bp;

    Bp->prev = B->prev; Bp->next = B;
    if(B->prev) B->prev->next = Bp;
    B->prev = Bp;

    for(int v=B->head; v!=-1; ){
        int nv = next[v];
        if(marked[v]){
            lb_detach(B, v, next, prev, owner);
            lb_push_back(Bp, v, next, prev, owner);
        }
        v = nv;
    }
    if(Bp->head==-1){
        lb_remove_block(Bp); free(Bp); Bp=NULL;
    }
    return Bp;
}


static void lbfs_order(const Graph *G, int *order) {
    int n = G->n;
    int *next = (int*)malloc(n * sizeof(int));
    int *prev = (int*)malloc(n * sizeof(int));
    LBBlock **owner = (LBBlock**)malloc(n * sizeof(LBBlock*));
    int *mark = (int*)calloc(n, sizeof(int));
    for (int i = 0; i < n; i++){ next[i] = -1; prev[i] = -1; owner[i] = NULL; }
    LBBlock **touched = (LBBlock**)malloc(n * sizeof(LBBlock*));
    int tcount = 0;


    LBBlock head = { -1, -1, NULL, NULL, 0, NULL };
    LBBlock *B0 = lb_new_block();
    for (int v = 0; v < n; v++) lb_push_back(B0, v, next, prev, owner);
    head.next = B0; B0->prev = &head;

    for (int idx = n - 1; idx >= 0; idx--) {
        LBBlock *B = head.next;
        int s = B->head;
        order[idx] = s;

        for (int t = G->off[s]; t < G->off[s + 1]; t++) {
            int u = G->adj[t];
            mark[u] = 1;
            LBBlock *blk = owner[u];
            if (blk && !blk->touched) { 
                blk->touched = 1;
                touched[tcount++] = blk;
            }
        }


        for (int t = G->off[s]; t < G->off[s + 1]; t++) {
            int u = G->adj[t];
            if (!mark[u] || !owner[u]) continue;
            LBBlock *blk = owner[u];
            LBBlock *Bp = blk->newblk;

            if (!Bp) {
                Bp = lb_new_block();
                blk->newblk = Bp;
                Bp->prev = blk->prev; Bp->next = blk;
                if (blk->prev) blk->prev->next = Bp;
                blk->prev = Bp;
                if (head.next == blk) head.next = Bp;
                if (!blk->touched) {
                    blk->touched = 1;
                    touched[tcount++] = blk;
                }
            }
            lb_detach(blk, u, next, prev, owner);
            lb_push_back(Bp, u, next, prev, owner);
        }

        LBBlock **to_free = (LBBlock**)malloc((tcount > 0 ? tcount : 1) * sizeof(LBBlock*));
        int fcount = 0;

        for (int i = 0; i < tcount; i++) {
            LBBlock *blk = touched[i];
            LBBlock *Bp  = blk->newblk;

            blk->touched = 0;
            blk->newblk  = NULL;

            if (Bp && Bp->head == -1) {
                if (Bp->prev) Bp->prev->next = Bp->next;
                if (Bp->next) Bp->next->prev = Bp->prev;
                if (head.next == Bp) head.next = Bp->next;
                free(Bp);
            }

            if (blk->head == -1) {
                if (blk->prev) blk->prev->next = blk->next;
                if (blk->next) blk->next->prev = blk->prev;
                if (head.next == blk) head.next = blk->next;
                to_free[fcount++] = blk;
            }
        }


        for (int j = 0; j < fcount; j++) free(to_free[j]);
        free(to_free);
        tcount = 0;

        lb_detach(B, s, next, prev, owner);
        if (B->head == -1) {
            if (B->prev) B->prev->next = B->next;
            if (B->next) B->next->prev = B->prev;
            if (head.next == B) head.next = B->next;
            free(B);
        }
    }

    free(touched);
    free(mark);
    free(owner);
    free(prev);
    free(next);
}




static void factorizing_permutation(const Graph *G, int *out){
    const int n = G->n;
    int *sigma = (int*)malloc(n*sizeof(int));
    lbfs_order(G, sigma);
    for(int i=0;i<n;i++) out[i]=sigma[i];
    free(sigma);
}



/* MD Tree*/

typedef enum { ND_LEAF=0, ND_SERIES=1, ND_PARALLEL=2, ND_PRIME=3 } Kind;

typedef struct Node Node;
struct Node {
    Kind kind;
    int v;
    int deg, cap;
    Node **ch;
};

static Node *node_new(Kind k){ Node *u=(Node*)calloc(1,sizeof(Node)); u->kind=k; u->v=-1; return u; }
static Node *node_leaf(int v){ Node *u=node_new(ND_LEAF); u->v=v; return u; }
static void node_add(Node *p, Node *c){
    if(p->deg==p->cap){ p->cap = p->cap? 2*p->cap : 4; p->ch=(Node**)realloc(p->ch,p->cap*sizeof(Node*)); }
    p->ch[p->deg++]=c;
}
static void node_free(Node *u){ if(!u) return; for(int i=0;i<u->deg;i++) node_free(u->ch[i]); free(u->ch); free(u); }

static void collect_leaves(Node *r, int *buf, int *k) {
    if (!r) return;
    if (r->kind == ND_LEAF) {
        buf[(*k)++] = r->v;
        return;
    }
    for (int i = 0; i < r->deg; i++) {
        collect_leaves(r->ch[i], buf, k);
    }
}




static int verify_modules(const Graph *G, Node *root){
    if(!root) return 1;
    int n=G->n; int *buf=(int*)malloc(n*sizeof(int)); unsigned char *in=(unsigned char*)calloc(n,1);
    Node **st=(Node**)malloc((2*n+5)*sizeof(Node*)); int top=0; st[top++]=root;
    int ok=1;
    while(top && ok){
        Node *u=st[--top];
        if(u->kind!=ND_LEAF){
            int k=0; collect_leaves(u,buf,&k);
            for(int i=0;i<k;i++) in[buf[i]]=1;
            for(int x=0;x<n;x++) if(!in[x]){
                int cnt=0;
                for(int t=G->off[x]; t<G->off[x+1]; t++) if(in[G->adj[t]]) cnt++;
                if(!(cnt==0 || cnt==k)){ ok=0; break; }
            }
            for(int i=0;i<k;i++) in[buf[i]]=0;
            for(int i=0;i<u->deg;i++) st[top++]=u->ch[i];
        }
    }
    free(st); free(in); free(buf);
    return ok;
}

/* Assembly */

typedef struct {
    Node *node;
    int *leaves; int len;
    int rep;
    int req_out; 
} BlockState;

static Node *assemble_md_incremental(const Graph *G, const int *perm){
    int n = G->n; 
    if (n == 0) return NULL;


    unsigned char *color_val = (unsigned char*)calloc(n, 1);
    unsigned int  *color_epoch = (unsigned int*)calloc(n, sizeof(unsigned int));
    unsigned char *mark_val = (unsigned char*)calloc(n, 1);
    unsigned int  *mark_epoch = (unsigned int*)calloc(n, sizeof(unsigned int));
    unsigned int epoch_color = 1, epoch_mark = 1;

    BlockState *stack = (BlockState*)malloc(n * sizeof(BlockState)); 
    int top = 0;


    BlockState S;
    int v0 = perm[0];
    S.node = node_leaf(v0);
    S.leaves = (int*)malloc(sizeof(int)); 
    S.leaves[0] = v0; 
    S.len = 1; 
    S.rep = v0;


    epoch_mark++;
    for (int t = G->off[v0]; t < G->off[v0+1]; t++) {
        int u = G->adj[t];
        mark_val[u] = 1;
        mark_epoch[u] = epoch_mark;
    }
    S.req_out = DEG(v0);

    color_val[v0] = 1; 
    color_epoch[v0] = epoch_color;

    stack[top++] = S;

    for (int i = 1; i < n; i++) {
        int v = perm[i];
        BlockState B;
        B.node = node_leaf(v);
        B.leaves = (int*)malloc(sizeof(int)); 
        B.leaves[0] = v; 
        B.len = 1; 
        B.rep = v;
        B.req_out = DEG(v);

        color_val[v] = 2; 
        color_epoch[v] = epoch_color;

        int merged = 1;
        while (merged && top >= 1) {
            merged = 0;
            S = stack[top-1];

            long edgesSB = 0;
            for (int bi = 0; bi < B.len; bi++) {
                int w = B.leaves[bi];
                for (int t = G->off[w]; t < G->off[w+1]; t++) {
                    int x = G->adj[t];
                    if (color_epoch[x]==epoch_color && color_val[x]==1)
                        edgesSB++;
                }
            }
            long full = (long)S.len * (long)B.len;
            int cross = -1;
            if (edgesSB == 0) cross = 0;
            else if (edgesSB == full) cross = 1;
            else break;


            int RinB = 0, ok = 1;
            for (int bi = 0; bi < B.len; bi++) {
                int w = B.leaves[bi], hit = 0;
                for (int t = G->off[w]; t < G->off[w+1]; t++) 
                    if (G->adj[t] == S.rep) { hit = 1; break; }
                if (hit) RinB++;
            }
            int expected = S.req_out - RinB; 
            if (expected < 0) ok = 0;

            for (int bi = 0; ok && bi < B.len; bi++) {
                int w = B.leaves[bi], dout = 0;
                for (int t = G->off[w]; t < G->off[w+1]; t++) {
                    int x = G->adj[t];
                    int c = (color_epoch[x]==epoch_color ? color_val[x] : 0);
                    if (c==0) {
                        if (!(mark_epoch[x]==epoch_mark && mark_val[x]==1)) { ok=0; break; }
                        dout++;
                    }
                }
                if (ok && dout != expected) ok = 0;
            }
            if (!ok) break;


            Node *L = S.node, *R = B.node; 
            if (!L || !R) break;
            if (cross==1) {
                if (L->kind==ND_SERIES && L->v==-1) node_add(L,R);
                else { Node *p=node_new(ND_SERIES); node_add(p,L); node_add(p,R); S.node=p; }
            } else {
                if (L->kind==ND_PARALLEL && L->v==-1) node_add(L,R);
                else { Node *p=node_new(ND_PARALLEL); node_add(p,L); node_add(p,R); S.node=p; }
            }
            for (int bi = 0; bi < B.len; bi++) {
                int w = B.leaves[bi];
                color_val[w] = 1; 
                color_epoch[w] = epoch_color;
            }
            S.leaves = (int*)realloc(S.leaves,(S.len+B.len)*sizeof(int));
            memcpy(S.leaves+S.len, B.leaves, B.len*sizeof(int));
            S.len += B.len;
            S.req_out -= RinB;

            stack[top-1] = S;
            B.node = NULL;
            merged = 1;
        }

        free(B.leaves);

        if (B.node != NULL) {
            epoch_color++;
            epoch_mark++;

            BlockState newS;
            newS.node = node_leaf(v);
            newS.leaves = (int*)malloc(sizeof(int));
            newS.leaves[0] = v;
            newS.len = 1;
            newS.rep = v;

            for (int t = G->off[v]; t < G->off[v+1]; t++) {
                int u = G->adj[t];
                mark_val[u] = 1;
                mark_epoch[u] = epoch_mark;
            }
            newS.req_out = DEG(v);

            color_val[v] = 1;
            color_epoch[v] = epoch_color;

            stack[top++] = newS;
        }

    }

    Node *root=NULL;
    if (top==1) root=stack[0].node;
    else { root=node_new(ND_PRIME); for(int i=0;i<top;i++) node_add(root,stack[i].node); }
    for (int i=0;i<top;i++) free(stack[i].leaves);
    free(stack); 
    free(mark_val); free(mark_epoch); 
    free(color_val); free(color_epoch);
    return root;
}


/* Printing and Demo*/

static int cmp_int(const void *a,const void *b){ int x=*(const int*)a, y=*(const int*)b; return (x>y)-(x<y); }
static void print_tree(Node *u, int n, int ind){
    for(int i=0;i<ind;i++) putchar(' ');
    if(u->kind==ND_LEAF){ printf("%d\n", u->v); return; }
    const char *K = (u->kind==ND_SERIES) ? "SERIES" : (u->kind==ND_PARALLEL) ? "PARALLEL" : "PRIME";
    int *buf=(int*)malloc(n*sizeof(int)); int k=0; collect_leaves(u,buf,&k); qsort(buf,k,sizeof(int),cmp_int);
    printf("("); for(int i=0;i<k;i++){ if(i) printf(", "); printf("%d", buf[i]); } printf(") %s\n", K);
    for(int i=0;i<u->deg;i++) print_tree(u->ch[i], n, ind+2);
    free(buf);
}

/* Main (with six graph examples) */


Node *modular_decomposition(Graph *G) {
    int n = G->n;
    int *sigma = (int*)malloc(n * sizeof(int));
    if (!sigma) { perror("malloc"); exit(1); }

    lbfs_order(G, sigma);                     /* O(n + m) */
    Node *root = assemble_md_incremental(G, sigma);

    free(sigma);
    return root;
}



int main(void) {
    /* Clique K4 */
    {
        int n = 4, deg[4] = {3,3,3,3};
        int L0[3] = {1,2,3}, L1[3] = {0,2,3}, L2[3] = {0,1,3}, L3[3] = {0,1,2};
        int *lists[4] = {L0,L1,L2,L3};
        Graph *G = graph_build(n, deg, lists);

        Node *root = modular_decomposition(G);
        printf("-- Clique4 --\n");
        print_tree(root, n, 0);
        printf("verify: %s\n\n", verify_modules(G, root) ? "True" : "False");

        node_free(root);
        graph_free(G);
    }

    /* K{3,2} */
    {
        int n = 5, deg[5] = {2,2,2,3,3};
        int L0[2] = {3,4}, L1[2] = {3,4}, L2[2] = {3,4}, L3[3] = {0,1,2}, L4[3] = {0,1,2};
        int *lists[5] = {L0,L1,L2,L3,L4};
        Graph *G = graph_build(n, deg, lists);

        Node *root = modular_decomposition(G);
        printf("-- K32 --\n");
        print_tree(root, n, 0);
        printf("verify: %s\n\n", verify_modules(G, root) ? "True" : "False");

        node_free(root);
        graph_free(G);
    }

    /* Tree */
    {
        int n = 5, deg[5] = {2,3,1,1,1};
        int L0[2] = {1,2}, L1[3] = {0,3,4}, L2[1] = {0}, L3[1] = {1}, L4[1] = {1};
        int *lists[5] = {L0,L1,L2,L3,L4};
        Graph *G = graph_build(n, deg, lists);

        Node *root = modular_decomposition(G);
        printf("-- Tree --\n");
        print_tree(root, n, 0);
        printf("verify: %s\n\n", verify_modules(G, root) ? "True" : "False");

        node_free(root);
        graph_free(G);
    }

    /* Nested */
    {
        int n = 5, deg[5] = {2,2,4,2,2};   // was {2,3,4,2,2} ← mismatch caused the crash
        int L0[2] = {1,2}, L1[2] = {0,2}, L2[4] = {0,1,3,4}, L3[2] = {2,4}, L4[2] = {2,3};
        int *lists[5] = {L0,L1,L2,L3,L4};
        Graph *G = graph_build(n, deg, lists);

        Node *root = modular_decomposition(G);
        printf("-- Nested --\n");
        print_tree(root, n, 0);
        printf("verify: %s\n\n", verify_modules(G, root) ? "True" : "False");

        node_free(root);
        graph_free(G);
    }

    /* C5 */
    {
        int n = 5, deg[5] = {2,2,2,2,2};
        int L0[2] = {1,4}, L1[2] = {0,2}, L2[2] = {1,3}, L3[2] = {2,4}, L4[2] = {3,0};
        int *lists[5] = {L0,L1,L2,L3,L4};
        Graph *G = graph_build(n, deg, lists);

        Node *root = modular_decomposition(G);
        printf("-- C5 --\n");
        print_tree(root, n, 0);
        printf("verify: %s\n\n", verify_modules(G, root) ? "True" : "False");

        node_free(root);
        graph_free(G);
    }

    /* K3 + K2 */
    {
        int n = 5, deg[5] = {2,2,2,1,1};
        int L0[2] = {1,2}, L1[2] = {0,2}, L2[2] = {0,1}, L3[1] = {4}, L4[1] = {3};
        int *lists[5] = {L0,L1,L2,L3,L4};
        Graph *G = graph_build(n, deg, lists);

        Node *root = modular_decomposition(G);
        printf("-- DisjointCliques --\n");
        print_tree(root, n, 0);
        printf("verify: %s\n\n", verify_modules(G, root) ? "True" : "False");

        node_free(root);
        graph_free(G);
    }

    return 0;
}


/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) solver_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

/* Add prefix to internal symbols */
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_f1 CASADI_PREFIX(f1)
#define casadi_f2 CASADI_PREFIX(f2)
#define casadi_fill CASADI_PREFIX(fill)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

void casadi_copy(const casadi_real* x, casadi_int n, casadi_real* y) {
  casadi_int i;
  if (y) {
    if (x) {
      for (i=0; i<n; ++i) *y++ = *x++;
    } else {
      for (i=0; i<n; ++i) *y++ = 0.;
    }
  }
}

void casadi_clear(casadi_real* x, casadi_int n) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = 0;
  }
}

#ifndef casadi_inf
  #define casadi_inf INFINITY
#endif

void casadi_fill(casadi_real* x, casadi_int n, casadi_real alpha) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = alpha;
  }
}

static const casadi_int casadi_s0[17] = {13, 1, 0, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
static const casadi_int casadi_s1[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s2[10] = {6, 1, 0, 6, 0, 1, 2, 3, 4, 5};

/* solver:(x0[1913],p[27],lbx[1913],ubx[1913],lbg[2913],ubg[2913],lam_x0[1913],lam_g0[2913])->(x[1913],f,g[2913],lam_x[1913],lam_g[2913],lam_p[27]) */
static int casadi_f1(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  struct casadi_nlpsol_data d_nlp;
  struct casadi_nlpsol_prob p_nlp;
  d_nlp.prob = &p_nlp;
  p_nlp.nx = 1913;
  p_nlp.ng = 2913;
  p_nlp.np = 27;
  casadi_nlpsol_init(&d_nlp, &iw, &w);
  return 0;
}

/* helper:(i0[1913],i1[27],i2[2913])->(o0[6]) */
static int casadi_f2(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real *rr, *ss;
  casadi_real *w0=w+0, *w1=w+600;
  /* #0: @0 = input[0][1] */
  casadi_copy(arg[0] ? arg[0]+1313 : 0, 600, w0);
  /* #1: @1 = @0[:6] */
  for (rr=w1, ss=w0+0; ss!=w0+6; ss+=1) *rr++ = *ss;
  /* #2: output[0][0] = @1 */
  casadi_copy(w1, 6, res[0]);
  return 0;
}

/* solver:(i0[13],i1,i2[13])->(o0[6]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i;
  casadi_real **res1=res+1, *rr;
  const casadi_real **arg1=arg+3, *cs;
  casadi_real *w0=w+167456, *w1=w+169369, *w2=w+169382, w3, *w4=w+169396, *w5=w+169423, *w6=w+171336, *w7=w+173249, *w8=w+173262, *w9=w+173275, *w10=w+173288, *w11=w+173301, *w12=w+173314, *w13=w+173327, *w14=w+173340, *w15=w+173353, *w16=w+173366, *w17=w+173379, *w18=w+173392, *w19=w+173405, *w20=w+173418, *w21=w+173431, *w22=w+173444, *w23=w+173457, *w24=w+173470, *w25=w+173483, *w26=w+173496, *w27=w+173509, *w28=w+173522, *w29=w+173535, *w30=w+173548, *w31=w+173561, *w32=w+173574, *w33=w+173587, *w34=w+173600, *w35=w+173613, *w36=w+173626, *w37=w+173639, *w38=w+173652, *w39=w+173665, *w40=w+173678, *w41=w+173691, *w42=w+173704, *w43=w+173717, *w44=w+173730, *w45=w+173743, *w46=w+173756, *w47=w+173769, *w48=w+173782, *w49=w+173795, *w50=w+173808, *w51=w+173821, *w52=w+173834, *w53=w+173847, *w54=w+173860, *w55=w+173873, *w56=w+173886, *w57=w+173899, *w58=w+173912, *w59=w+173925, *w60=w+173938, *w61=w+173951, *w62=w+173964, *w63=w+173977, *w64=w+173990, *w65=w+174003, *w66=w+174016, *w67=w+174029, *w68=w+174042, *w69=w+174055, *w70=w+174068, *w71=w+174081, *w72=w+174094, *w73=w+174107, *w74=w+174120, *w75=w+174133, *w76=w+174146, *w77=w+174159, *w78=w+174172, *w79=w+174185, *w80=w+174198, *w81=w+174211, *w82=w+174224, *w83=w+174237, *w84=w+174250, *w85=w+174263, *w86=w+174276, *w87=w+174289, *w88=w+174302, *w89=w+174315, *w90=w+174328, *w91=w+174341, *w92=w+174354, *w93=w+174367, *w94=w+174380, *w95=w+174393, *w96=w+174406, *w97=w+174419, *w98=w+174432, *w99=w+174445, *w100=w+174458, *w101=w+174471, *w102=w+174484, *w103=w+174497, *w104=w+174510, *w105=w+174523, *w106=w+174536, *w107=w+174636, *w108=w+174736, *w109=w+174836, *w110=w+174936, *w111=w+175036, *w112=w+175136, *w113=w+175236, *w114=w+175336, *w115=w+175436, *w116=w+175536, *w117=w+175636, *w118=w+175736, *w119=w+175836, *w120=w+175936, *w121=w+176036, *w122=w+176136, *w123=w+179049, *w124=w+181962, *w125=w+183875, *w126=w+186788, *w127=w+188701, *w128=w+191614;
  /* #0: @0 = zeros(1913x1) */
  casadi_clear(w0, 1913);
  /* #1: @1 = input[0][0] */
  casadi_copy(arg[0], 13, w1);
  /* #2: @2 = input[2][0] */
  casadi_copy(arg[2], 13, w2);
  /* #3: @3 = input[1][0] */
  w3 = arg[1] ? arg[1][0] : 0;
  /* #4: @4 = vertcat(@1, @2, @3) */
  rr=w4;
  for (i=0, cs=w1; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w2; i<13; ++i) *rr++ = *cs++;
  *rr++ = w3;
  /* #5: @5 = -inf(1913x1) */
  casadi_fill(w5, 1913, -casadi_inf);
  /* #6: @6 = inf(1913x1) */
  casadi_fill(w6, 1913, casadi_inf);
  /* #7: @2 = zeros(13x1) */
  casadi_clear(w2, 13);
  /* #8: @7 = zeros(13x1) */
  casadi_clear(w7, 13);
  /* #9: @8 = zeros(13x1) */
  casadi_clear(w8, 13);
  /* #10: @9 = zeros(13x1) */
  casadi_clear(w9, 13);
  /* #11: @10 = zeros(13x1) */
  casadi_clear(w10, 13);
  /* #12: @11 = zeros(13x1) */
  casadi_clear(w11, 13);
  /* #13: @12 = zeros(13x1) */
  casadi_clear(w12, 13);
  /* #14: @13 = zeros(13x1) */
  casadi_clear(w13, 13);
  /* #15: @14 = zeros(13x1) */
  casadi_clear(w14, 13);
  /* #16: @15 = zeros(13x1) */
  casadi_clear(w15, 13);
  /* #17: @16 = zeros(13x1) */
  casadi_clear(w16, 13);
  /* #18: @17 = zeros(13x1) */
  casadi_clear(w17, 13);
  /* #19: @18 = zeros(13x1) */
  casadi_clear(w18, 13);
  /* #20: @19 = zeros(13x1) */
  casadi_clear(w19, 13);
  /* #21: @20 = zeros(13x1) */
  casadi_clear(w20, 13);
  /* #22: @21 = zeros(13x1) */
  casadi_clear(w21, 13);
  /* #23: @22 = zeros(13x1) */
  casadi_clear(w22, 13);
  /* #24: @23 = zeros(13x1) */
  casadi_clear(w23, 13);
  /* #25: @24 = zeros(13x1) */
  casadi_clear(w24, 13);
  /* #26: @25 = zeros(13x1) */
  casadi_clear(w25, 13);
  /* #27: @26 = zeros(13x1) */
  casadi_clear(w26, 13);
  /* #28: @27 = zeros(13x1) */
  casadi_clear(w27, 13);
  /* #29: @28 = zeros(13x1) */
  casadi_clear(w28, 13);
  /* #30: @29 = zeros(13x1) */
  casadi_clear(w29, 13);
  /* #31: @30 = zeros(13x1) */
  casadi_clear(w30, 13);
  /* #32: @31 = zeros(13x1) */
  casadi_clear(w31, 13);
  /* #33: @32 = zeros(13x1) */
  casadi_clear(w32, 13);
  /* #34: @33 = zeros(13x1) */
  casadi_clear(w33, 13);
  /* #35: @34 = zeros(13x1) */
  casadi_clear(w34, 13);
  /* #36: @35 = zeros(13x1) */
  casadi_clear(w35, 13);
  /* #37: @36 = zeros(13x1) */
  casadi_clear(w36, 13);
  /* #38: @37 = zeros(13x1) */
  casadi_clear(w37, 13);
  /* #39: @38 = zeros(13x1) */
  casadi_clear(w38, 13);
  /* #40: @39 = zeros(13x1) */
  casadi_clear(w39, 13);
  /* #41: @40 = zeros(13x1) */
  casadi_clear(w40, 13);
  /* #42: @41 = zeros(13x1) */
  casadi_clear(w41, 13);
  /* #43: @42 = zeros(13x1) */
  casadi_clear(w42, 13);
  /* #44: @43 = zeros(13x1) */
  casadi_clear(w43, 13);
  /* #45: @44 = zeros(13x1) */
  casadi_clear(w44, 13);
  /* #46: @45 = zeros(13x1) */
  casadi_clear(w45, 13);
  /* #47: @46 = zeros(13x1) */
  casadi_clear(w46, 13);
  /* #48: @47 = zeros(13x1) */
  casadi_clear(w47, 13);
  /* #49: @48 = zeros(13x1) */
  casadi_clear(w48, 13);
  /* #50: @49 = zeros(13x1) */
  casadi_clear(w49, 13);
  /* #51: @50 = zeros(13x1) */
  casadi_clear(w50, 13);
  /* #52: @51 = zeros(13x1) */
  casadi_clear(w51, 13);
  /* #53: @52 = zeros(13x1) */
  casadi_clear(w52, 13);
  /* #54: @53 = zeros(13x1) */
  casadi_clear(w53, 13);
  /* #55: @54 = zeros(13x1) */
  casadi_clear(w54, 13);
  /* #56: @55 = zeros(13x1) */
  casadi_clear(w55, 13);
  /* #57: @56 = zeros(13x1) */
  casadi_clear(w56, 13);
  /* #58: @57 = zeros(13x1) */
  casadi_clear(w57, 13);
  /* #59: @58 = zeros(13x1) */
  casadi_clear(w58, 13);
  /* #60: @59 = zeros(13x1) */
  casadi_clear(w59, 13);
  /* #61: @60 = zeros(13x1) */
  casadi_clear(w60, 13);
  /* #62: @61 = zeros(13x1) */
  casadi_clear(w61, 13);
  /* #63: @62 = zeros(13x1) */
  casadi_clear(w62, 13);
  /* #64: @63 = zeros(13x1) */
  casadi_clear(w63, 13);
  /* #65: @64 = zeros(13x1) */
  casadi_clear(w64, 13);
  /* #66: @65 = zeros(13x1) */
  casadi_clear(w65, 13);
  /* #67: @66 = zeros(13x1) */
  casadi_clear(w66, 13);
  /* #68: @67 = zeros(13x1) */
  casadi_clear(w67, 13);
  /* #69: @68 = zeros(13x1) */
  casadi_clear(w68, 13);
  /* #70: @69 = zeros(13x1) */
  casadi_clear(w69, 13);
  /* #71: @70 = zeros(13x1) */
  casadi_clear(w70, 13);
  /* #72: @71 = zeros(13x1) */
  casadi_clear(w71, 13);
  /* #73: @72 = zeros(13x1) */
  casadi_clear(w72, 13);
  /* #74: @73 = zeros(13x1) */
  casadi_clear(w73, 13);
  /* #75: @74 = zeros(13x1) */
  casadi_clear(w74, 13);
  /* #76: @75 = zeros(13x1) */
  casadi_clear(w75, 13);
  /* #77: @76 = zeros(13x1) */
  casadi_clear(w76, 13);
  /* #78: @77 = zeros(13x1) */
  casadi_clear(w77, 13);
  /* #79: @78 = zeros(13x1) */
  casadi_clear(w78, 13);
  /* #80: @79 = zeros(13x1) */
  casadi_clear(w79, 13);
  /* #81: @80 = zeros(13x1) */
  casadi_clear(w80, 13);
  /* #82: @81 = zeros(13x1) */
  casadi_clear(w81, 13);
  /* #83: @82 = zeros(13x1) */
  casadi_clear(w82, 13);
  /* #84: @83 = zeros(13x1) */
  casadi_clear(w83, 13);
  /* #85: @84 = zeros(13x1) */
  casadi_clear(w84, 13);
  /* #86: @85 = zeros(13x1) */
  casadi_clear(w85, 13);
  /* #87: @86 = zeros(13x1) */
  casadi_clear(w86, 13);
  /* #88: @87 = zeros(13x1) */
  casadi_clear(w87, 13);
  /* #89: @88 = zeros(13x1) */
  casadi_clear(w88, 13);
  /* #90: @89 = zeros(13x1) */
  casadi_clear(w89, 13);
  /* #91: @90 = zeros(13x1) */
  casadi_clear(w90, 13);
  /* #92: @91 = zeros(13x1) */
  casadi_clear(w91, 13);
  /* #93: @92 = zeros(13x1) */
  casadi_clear(w92, 13);
  /* #94: @93 = zeros(13x1) */
  casadi_clear(w93, 13);
  /* #95: @94 = zeros(13x1) */
  casadi_clear(w94, 13);
  /* #96: @95 = zeros(13x1) */
  casadi_clear(w95, 13);
  /* #97: @96 = zeros(13x1) */
  casadi_clear(w96, 13);
  /* #98: @97 = zeros(13x1) */
  casadi_clear(w97, 13);
  /* #99: @98 = zeros(13x1) */
  casadi_clear(w98, 13);
  /* #100: @99 = zeros(13x1) */
  casadi_clear(w99, 13);
  /* #101: @100 = zeros(13x1) */
  casadi_clear(w100, 13);
  /* #102: @101 = zeros(13x1) */
  casadi_clear(w101, 13);
  /* #103: @102 = zeros(13x1) */
  casadi_clear(w102, 13);
  /* #104: @103 = zeros(13x1) */
  casadi_clear(w103, 13);
  /* #105: @104 = zeros(13x1) */
  casadi_clear(w104, 13);
  /* #106: @105 = zeros(13x1) */
  casadi_clear(w105, 13);
  /* #107: @106 = -inf(100x1) */
  casadi_fill(w106, 100, -casadi_inf);
  /* #108: @107 = -inf(100x1) */
  casadi_fill(w107, 100, -casadi_inf);
  /* #109: @108 = -inf(100x1) */
  casadi_fill(w108, 100, -casadi_inf);
  /* #110: @109 = -inf(100x1) */
  casadi_fill(w109, 100, -casadi_inf);
  /* #111: @110 = -inf(100x1) */
  casadi_fill(w110, 100, -casadi_inf);
  /* #112: @111 = -inf(100x1) */
  casadi_fill(w111, 100, -casadi_inf);
  /* #113: @112 = -inf(100x1) */
  casadi_fill(w112, 100, -casadi_inf);
  /* #114: @113 = -inf(100x1) */
  casadi_fill(w113, 100, -casadi_inf);
  /* #115: @114 = -inf(100x1) */
  casadi_fill(w114, 100, -casadi_inf);
  /* #116: @115 = -inf(100x1) */
  casadi_fill(w115, 100, -casadi_inf);
  /* #117: @116 = -inf(100x1) */
  casadi_fill(w116, 100, -casadi_inf);
  /* #118: @117 = -inf(100x1) */
  casadi_fill(w117, 100, -casadi_inf);
  /* #119: @118 = -inf(100x1) */
  casadi_fill(w118, 100, -casadi_inf);
  /* #120: @119 = -inf(100x1) */
  casadi_fill(w119, 100, -casadi_inf);
  /* #121: @120 = -inf(100x1) */
  casadi_fill(w120, 100, -casadi_inf);
  /* #122: @121 = -inf(100x1) */
  casadi_fill(w121, 100, -casadi_inf);
  /* #123: @122 = vertcat(@1, @2, @7, @8, @9, @10, @11, @12, @13, @14, @15, @16, @17, @18, @19, @20, @21, @22, @23, @24, @25, @26, @27, @28, @29, @30, @31, @32, @33, @34, @35, @36, @37, @38, @39, @40, @41, @42, @43, @44, @45, @46, @47, @48, @49, @50, @51, @52, @53, @54, @55, @56, @57, @58, @59, @60, @61, @62, @63, @64, @65, @66, @67, @68, @69, @70, @71, @72, @73, @74, @75, @76, @77, @78, @79, @80, @81, @82, @83, @84, @85, @86, @87, @88, @89, @90, @91, @92, @93, @94, @95, @96, @97, @98, @99, @100, @101, @102, @103, @104, @105, @106, @107, @108, @109, @110, @111, @112, @113, @114, @115, @116, @117, @118, @119, @120, @121) */
  rr=w122;
  for (i=0, cs=w1; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w2; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w7; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w8; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w9; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w10; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w11; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w12; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w13; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w14; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w15; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w16; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w17; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w18; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w19; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w20; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w21; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w22; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w23; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w24; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w25; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w26; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w27; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w28; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w29; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w30; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w31; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w32; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w33; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w34; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w35; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w36; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w37; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w38; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w39; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w40; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w41; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w42; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w43; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w44; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w45; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w46; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w47; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w48; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w49; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w50; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w51; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w52; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w53; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w54; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w55; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w56; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w57; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w58; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w59; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w60; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w61; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w62; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w63; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w64; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w65; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w66; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w67; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w68; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w69; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w70; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w71; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w72; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w73; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w74; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w75; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w76; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w77; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w78; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w79; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w80; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w81; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w82; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w83; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w84; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w85; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w86; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w87; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w88; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w89; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w90; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w91; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w92; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w93; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w94; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w95; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w96; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w97; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w98; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w99; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w100; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w101; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w102; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w103; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w104; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w105; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w106; i<100; ++i) *rr++ = *cs++;
  for (i=0, cs=w107; i<100; ++i) *rr++ = *cs++;
  for (i=0, cs=w108; i<100; ++i) *rr++ = *cs++;
  for (i=0, cs=w109; i<100; ++i) *rr++ = *cs++;
  for (i=0, cs=w110; i<100; ++i) *rr++ = *cs++;
  for (i=0, cs=w111; i<100; ++i) *rr++ = *cs++;
  for (i=0, cs=w112; i<100; ++i) *rr++ = *cs++;
  for (i=0, cs=w113; i<100; ++i) *rr++ = *cs++;
  for (i=0, cs=w114; i<100; ++i) *rr++ = *cs++;
  for (i=0, cs=w115; i<100; ++i) *rr++ = *cs++;
  for (i=0, cs=w116; i<100; ++i) *rr++ = *cs++;
  for (i=0, cs=w117; i<100; ++i) *rr++ = *cs++;
  for (i=0, cs=w118; i<100; ++i) *rr++ = *cs++;
  for (i=0, cs=w119; i<100; ++i) *rr++ = *cs++;
  for (i=0, cs=w120; i<100; ++i) *rr++ = *cs++;
  for (i=0, cs=w121; i<100; ++i) *rr++ = *cs++;
  /* #124: @106 = zeros(100x1) */
  casadi_clear(w106, 100);
  /* #125: @107 = zeros(100x1) */
  casadi_clear(w107, 100);
  /* #126: @108 = zeros(100x1) */
  casadi_clear(w108, 100);
  /* #127: @109 = zeros(100x1) */
  casadi_clear(w109, 100);
  /* #128: @110 = zeros(100x1) */
  casadi_clear(w110, 100);
  /* #129: @111 = zeros(100x1) */
  casadi_clear(w111, 100);
  /* #130: @112 = zeros(100x1) */
  casadi_clear(w112, 100);
  /* #131: @113 = zeros(100x1) */
  casadi_clear(w113, 100);
  /* #132: @114 = zeros(100x1) */
  casadi_clear(w114, 100);
  /* #133: @115 = zeros(100x1) */
  casadi_clear(w115, 100);
  /* #134: @116 = zeros(100x1) */
  casadi_clear(w116, 100);
  /* #135: @117 = zeros(100x1) */
  casadi_clear(w117, 100);
  /* #136: @118 = zeros(100x1) */
  casadi_clear(w118, 100);
  /* #137: @119 = zeros(100x1) */
  casadi_clear(w119, 100);
  /* #138: @120 = zeros(100x1) */
  casadi_clear(w120, 100);
  /* #139: @121 = zeros(100x1) */
  casadi_clear(w121, 100);
  /* #140: @123 = vertcat(@1, @2, @7, @8, @9, @10, @11, @12, @13, @14, @15, @16, @17, @18, @19, @20, @21, @22, @23, @24, @25, @26, @27, @28, @29, @30, @31, @32, @33, @34, @35, @36, @37, @38, @39, @40, @41, @42, @43, @44, @45, @46, @47, @48, @49, @50, @51, @52, @53, @54, @55, @56, @57, @58, @59, @60, @61, @62, @63, @64, @65, @66, @67, @68, @69, @70, @71, @72, @73, @74, @75, @76, @77, @78, @79, @80, @81, @82, @83, @84, @85, @86, @87, @88, @89, @90, @91, @92, @93, @94, @95, @96, @97, @98, @99, @100, @101, @102, @103, @104, @105, @106, @107, @108, @109, @110, @111, @112, @113, @114, @115, @116, @117, @118, @119, @120, @121) */
  rr=w123;
  for (i=0, cs=w1; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w2; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w7; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w8; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w9; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w10; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w11; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w12; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w13; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w14; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w15; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w16; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w17; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w18; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w19; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w20; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w21; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w22; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w23; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w24; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w25; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w26; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w27; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w28; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w29; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w30; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w31; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w32; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w33; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w34; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w35; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w36; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w37; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w38; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w39; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w40; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w41; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w42; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w43; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w44; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w45; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w46; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w47; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w48; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w49; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w50; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w51; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w52; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w53; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w54; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w55; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w56; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w57; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w58; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w59; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w60; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w61; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w62; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w63; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w64; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w65; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w66; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w67; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w68; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w69; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w70; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w71; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w72; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w73; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w74; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w75; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w76; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w77; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w78; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w79; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w80; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w81; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w82; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w83; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w84; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w85; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w86; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w87; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w88; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w89; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w90; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w91; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w92; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w93; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w94; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w95; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w96; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w97; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w98; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w99; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w100; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w101; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w102; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w103; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w104; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w105; i<13; ++i) *rr++ = *cs++;
  for (i=0, cs=w106; i<100; ++i) *rr++ = *cs++;
  for (i=0, cs=w107; i<100; ++i) *rr++ = *cs++;
  for (i=0, cs=w108; i<100; ++i) *rr++ = *cs++;
  for (i=0, cs=w109; i<100; ++i) *rr++ = *cs++;
  for (i=0, cs=w110; i<100; ++i) *rr++ = *cs++;
  for (i=0, cs=w111; i<100; ++i) *rr++ = *cs++;
  for (i=0, cs=w112; i<100; ++i) *rr++ = *cs++;
  for (i=0, cs=w113; i<100; ++i) *rr++ = *cs++;
  for (i=0, cs=w114; i<100; ++i) *rr++ = *cs++;
  for (i=0, cs=w115; i<100; ++i) *rr++ = *cs++;
  for (i=0, cs=w116; i<100; ++i) *rr++ = *cs++;
  for (i=0, cs=w117; i<100; ++i) *rr++ = *cs++;
  for (i=0, cs=w118; i<100; ++i) *rr++ = *cs++;
  for (i=0, cs=w119; i<100; ++i) *rr++ = *cs++;
  for (i=0, cs=w120; i<100; ++i) *rr++ = *cs++;
  for (i=0, cs=w121; i<100; ++i) *rr++ = *cs++;
  /* #141: @124 = zeros(1913x1) */
  casadi_clear(w124, 1913);
  /* #142: @125 = zeros(2913x1) */
  casadi_clear(w125, 2913);
  /* #143: {@126, NULL, NULL, NULL, @127, NULL} = solver(@0, @4, @5, @6, @122, @123, @124, @125) */
  arg1[0]=w0;
  arg1[1]=w4;
  arg1[2]=w5;
  arg1[3]=w6;
  arg1[4]=w122;
  arg1[5]=w123;
  arg1[6]=w124;
  arg1[7]=w125;
  res1[0]=w126;
  res1[1]=0;
  res1[2]=0;
  res1[3]=0;
  res1[4]=w127;
  res1[5]=0;
  if (casadi_f1(arg1, res1, iw, w, 0)) return 1;
  /* #144: @128 = helper(@126, @4, @127) */
  arg1[0]=w126;
  arg1[1]=w4;
  arg1[2]=w127;
  res1[0]=w128;
  if (casadi_f2(arg1, res1, iw, w, 0)) return 1;
  /* #145: output[0][0] = @128 */
  casadi_copy(w128, 6, res[0]);
  return 0;
}

CASADI_SYMBOL_EXPORT int solver(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int solver_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int solver_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void solver_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int solver_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void solver_release(int mem) {
}

CASADI_SYMBOL_EXPORT void solver_incref(void) {
}

CASADI_SYMBOL_EXPORT void solver_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int solver_n_in(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_int solver_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real solver_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* solver_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* solver_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* solver_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* solver_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int solver_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 130;
  if (sz_res) *sz_res = 128;
  if (sz_iw) *sz_iw = 1914;
  if (sz_w) *sz_w = 191620;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
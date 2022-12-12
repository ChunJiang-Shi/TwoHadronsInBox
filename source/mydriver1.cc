#include "box_quant.h"
#include "npy.h"

#include <fstream>
#include <vector>
#include <string>

#include <gsl/gsl_multimin.h>
#include <gsl/gsl_vector.h>
using namespace std;

//  copy from Example driver program 2.  This done exactly the same things
//  as "driver1.cc", but does not need an XML input program.
//  The quantities are built up in the code.

extern double mu = 8.0;

std::vector<double> readNpy(const std::string file)
{
  bool fortran_order = false;
  std::vector<unsigned long> shape{};
  std::vector<double> data;

  npy::LoadArrayFromNumpy(file, shape, fortran_order, data);
  return data;
}

struct fit_params
{
  BoxQuantization *BQ_pointer;
  vector<double> Ecm;
};

int print_state(size_t iter, gsl_multimin_fminimizer *s, double size)
{
  printf("iter = %3u x = % .3f % .3f % .3f % .3f % .3f % .3f ",
         iter,
         gsl_vector_get(s->x, 0),
         gsl_vector_get(s->x, 1),
         gsl_vector_get(s->x, 2),
         gsl_vector_get(s->x, 3),
         gsl_vector_get(s->x, 4),
         gsl_vector_get(s->x, 5));
  printf("\t\tf(x) = % .3e size = %.3e\n",
         s->fval, size);
}

double my_minimaize_f(const gsl_vector *fit_kappa, void *p)
{
  double ret = .0;
  struct fit_params *fp = (struct fit_params *)p;
  vector<double> Ecm_local = fp->Ecm;
  // const double tmp = static_cast<double>(gsl_vector_get(fit_kappa, 0));
  // printf("CBUF:XXX %e\n\n", tmp);
  vector<double> vec_kappa = {
      gsl_vector_get(fit_kappa, 0),
      gsl_vector_get(fit_kappa, 1),
      gsl_vector_get(fit_kappa, 2),
      gsl_vector_get(fit_kappa, 3),
      gsl_vector_get(fit_kappa, 4),
      gsl_vector_get(fit_kappa, 5),
  };
  // vec_kappa.reserve(6);

  fp->BQ_pointer->setKtildeParameters(vec_kappa);

  for (unsigned int i = 0; i < Ecm_local.size(); i++)
  {
    double Ecmi = Ecm_local[i];
    double omega = fp->BQ_pointer->getOmegaFromElab(mu, Ecmi);
    // printf("DBUG vec: %d, %e \t %e\n", i, omega, ret);
    ret += omega * omega;
  }
  return ret;
}

int main()
{
  // const std::string m1_file  = "/home/shi-wsl2/lqcd/TwoHadronsInBox/test/m1.npy";
  // const std::string m2_file  = "/home/shi-wsl2/lqcd/TwoHadronsInBox/test/m2.npy";
  const std::string m12_file = "/home/shi-wsl2/lqcd/TwoHadronsInBox/test/mass12_8.npy";

  std::vector<unsigned long> shape{};
  std::vector<double> m1_data;
  std::vector<double> m2_data;
  std::vector<double> m12_data;
  // m1_data  = readNpy( m1_file);
  // m2_data  = readNpy( m2_file);
  m12_data = readNpy(m12_file);
  // if (m1_data.size() != m2_data.size() || m2_data.size() != m12_data.size()){
  //      cout<<"Error: Input mass vactor not consistant!"<<m1_data.size()
  //                                              <<m2_data.size()
  //                                              <<m12_data.size()
  //                                              <<endl<<endl;
  //      throw(std::invalid_argument("Error: Input mass vactor not consistant!"));
  // }

  set<uint> powers;
  powers.insert(0);
  powers.insert(1);
  // powers.insert(2);
  // powers.insert(3);
  list<pair<KElementInfo, Polynomial>> pelems;
  pelems.push_back(make_pair(KElementInfo(2, 0, 2, 0, 0, 2, 0), Polynomial(powers)));
  pelems.push_back(make_pair(KElementInfo(2, 0, 2, 1, 0, 2, 0), Polynomial(powers)));
  pelems.push_back(make_pair(KElementInfo(2, 0, 2, 1, 0, 2, 1), Polynomial(powers)));
  KtildeInverseCalculator Kinv(pelems);
  cout << Kinv.output() << endl;
  cout << "Kinv #################################" << endl;

  string mom_ray("ar");
  uint mom_int_sq = 0;
  string lgirrep("T1g");
  vector<DecayChannelInfo> chan_infos;
  vector<uint> lmaxes;
  chan_infos.push_back(DecayChannelInfo("pion", "J/psi", 0, 2, false, true));
  chan_infos.push_back(DecayChannelInfo("D", "Dstar", 0, 2, false, true));
  lmaxes.push_back(0);
  lmaxes.push_back(0);

  BoxQuantization BQ(mom_ray, mom_int_sq, lgirrep, chan_infos, lmaxes, &Kinv);
  cout << "BQ BQ.output()" << endl;
  cout << BQ.output() << endl;
  cout << "BQ BQ.outputBasis(2)" << endl;
  cout << BQ.outputBasis(2) << endl;
  cout << "BQ #################################" << endl;

  double Elab_over_mref, Ecm, L_mref, mu, m1, m2;
  uint chan = 0;
  cout.precision(15);

  L_mref = 84.8;
  vector<double> residual = {
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
  };
  vector<double> kpara = {
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
  };
  BQ.setKtildeParameters(kpara);
  BQ.setRefMassL(L_mref);
  BQ.setMassesOverRef(0, 0.050352, 0.449531);
  BQ.setMassesOverRef(1, 0.273010, 0.293391);
  for (unsigned int ie = 0; ie < m12_data.size(); ie++)
  {
    Elab_over_mref = m12_data[ie];
    mu = 8.0;
    residual[ie] = BQ.getOmegaFromElab(mu, m12_data[ie]);

    // ComplexHermitianMatrix B;
    // RealSymmetricMatrix Kinv_output;
    // BQ.getBoxMatrixFromElab(Elab_over_mref, B);
    // BQ.getKtildeOrInverseFromEcm(Elab_over_mref, Kinv_output);
    cout << "     Omega = " << BQ.getOmegaFromElab(mu, Elab_over_mref) << endl
         << endl;
    cout << "       Ecm = " << Elab_over_mref << endl
         << endl;
    // // if (BQ.m_Kinv!=0){
    // //      cout << "BQ.m_Kinv!=0" <<endl;
    // // }
    // cout << "  Kinv_out = (" << Kinv_output.get(0,0) << Kinv_output.get(0,1) << ")\n"
    //      << "             (" << Kinv_output.get(1,0) << Kinv_output.get(1,1) << ")\n"
    //           <<endl<<endl;
    // cout << "         B = (" << B.get(0,0) << B.get(0,1) << ")\n"
    //      << "             (" << B.get(1,0) << B.get(1,1) << ")\n"
    //           <<endl<<endl;
    // double detroot=BQ.getDeterminantRoot(Kv,B, Ndet);
  }

  const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2rand;
  gsl_multimin_fminimizer *s = NULL;
  int status;
  size_t iter = 0;
  double size;
  const size_t dof = kpara.size();
  const size_t neq = m12_data.size();
  printf("dof = %d \t neq = %d\n", dof, neq);

  fit_params params = {
    BQ_pointer : &BQ,
    Ecm : m12_data,
  };
  gsl_multimin_function solverF;
  solverF.f = &my_minimaize_f;
  solverF.n = dof;
  solverF.params = &params;

  printf("DBUG: start:\n");

  double kappa_prior[dof]{1.0};

  gsl_vector *x = gsl_vector_alloc(dof);
  gsl_vector_set_all(x, 0.0);
  gsl_vector *step_size;
  step_size = gsl_vector_alloc(dof);
  gsl_vector_set_all(step_size, 1.0);

  s = gsl_multimin_fminimizer_alloc(T, dof);
  gsl_multimin_fminimizer_set(s, &solverF, x, step_size);

  printf("Start iter######################\n");

  print_state(iter, s, size);
  do
  {
    iter++;
    status = gsl_multimin_fminimizer_iterate(s);
    if (status) /* check if solver is stuck */
      break;
    size = gsl_multimin_fminimizer_size(s);
    status = gsl_multimin_test_size(size, 1e-2);
    print_state(iter, s, size);
  } while (status == GSL_CONTINUE && iter < 10000);

  printf("status = %s\n", gsl_strerror(status));

  if (status == GSL_SUCCESS)
  {
    for (size_t i = 0; i < dof; i++)
    {
      kpara[i] = gsl_vector_get(s->x, i);
    }
    BQ.setKtildeParameters(kpara);
    for (unsigned int ie = 0; ie < m12_data.size(); ie++)
    {
      ComplexHermitianMatrix B;
      RealSymmetricMatrix Kinv_output;
      Elab_over_mref = m12_data[ie];
      BQ.getBoxMatrixFromElab(Elab_over_mref, B);
      BQ.getKtildeOrInverseFromEcm(Elab_over_mref, Kinv_output);
      cout << "     Omega = " << BQ.getOmegaFromElab(mu, Elab_over_mref) << endl
           << endl;
      // // if (BQ.m_Kinv!=0){
      // //      cout << "BQ.m_Kinv!=0" <<endl;
      // // }
      cout << "  Kinv_out = (" << Kinv_output.get(0, 0) << Kinv_output.get(0, 1) << ")\n"
           << "             (" << Kinv_output.get(1, 0) << Kinv_output.get(1, 1) << ")\n"
           << endl
           << endl;
      cout << "         B = (" << B.get(0, 0) << B.get(0, 1) << ")\n"
           << "             (" << B.get(1, 0) << B.get(1, 1) << ")\n"
           << endl
           << endl;
      // double detroot=BQ.getDeterminantRoot(Kv,B, Ndet);
    }
  }
  else
  {
    cout << "---------Fit failed!-------------------" << endl
         << endl;
  }

  gsl_vector_free(x);
  gsl_vector_free(step_size);
  gsl_multimin_fminimizer_free(s);

  return 0;
}

// Minimalize the chi square fit for finite volume multi - channel scattering.
// input : Jackknife resampling data
// Author : Chunjiang Shi

#include "box_quant.h"
#include "npy.h"

#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <eigen3/Eigen/Dense>

#include <gsl/gsl_multimin.h>
#include <gsl/gsl_vector.h>
using namespace std;

#define DEGREE_OF_FREEDOM (12)

extern const double mu = 8.0;
extern const vector<double> blockList = {139, 144};
extern const vector<double> energyNumList = {8, 14};
extern const size_t lattNUM = 2;
extern const size_t offset_block[lattNUM] = {0, 139};
extern const size_t offset_e[lattNUM] = {0, 8};
extern const size_t offset_latt[lattNUM] = {0, 139 * 8};
extern const size_t globalSize = 139 * 8 + 144 * 14;
extern const size_t energyNumAll = std::accumulate(energyNumList.begin(), energyNumList.end(), 0);

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
  vector<double> *L_mref;
  vector<double> *Ecm;
  vector<double> *m1A;
  vector<double> *m1B;
  vector<double> *m2A;
  vector<double> *m2B;
};

int print_state(size_t iter, gsl_multimin_fminimizer *s, double size)
{
  printf("iter = %3u x = ", iter);
  for (size_t i = 0; i < DEGREE_OF_FREEDOM; i++)
    printf(" % .3f", gsl_vector_get(s->x, i));
  printf("\t\t f(x) = % .3e size = %.3e\n",
         s->fval, size);
}

double my_minimaize_f(const gsl_vector *fit_kappa, void *p)
{
  double ret = .0;
  struct fit_params *fp = (struct fit_params *)p;
  vector<double> vec_kappa;
  for (size_t i = 0; i < DEGREE_OF_FREEDOM; i++)
    vec_kappa.push_back(gsl_vector_get(fit_kappa, i));

  vector<double> ret_r;
  ret_r.reserve(fp->Ecm->size());

  fp->BQ_pointer->setKtildeParameters(vec_kappa);
  // #pragma omp parallel for reduction(+ \
                                   : ret)
  for (size_t ilat = 0; ilat < lattNUM; ilat++)
    for (size_t ib = 0; ib < blockList[ilat]; ib++)
    {
      size_t pos = offset_latt[ilat] + ib * energyNumList[ilat];
      fp->BQ_pointer->setRefMassL(fp->L_mref->at(pos));
      fp->BQ_pointer->setMassesOverRef(0, fp->m1A->at(pos), fp->m1B->at(pos));
      fp->BQ_pointer->setMassesOverRef(1, fp->m2A->at(pos), fp->m2B->at(pos));
      for (size_t ie = 0; ie < energyNumList[ilat]; ie++)
      {
        // printf("DBUG vec: ib = %d, m1A = %e, m1B = %e, m2A = %e, m2B = %e  \n", ib, (fp->m1A)[4 * ib], (fp->m1B)[4 * ib], (fp->m2A)[4 * ib], (fp->m2B)[4 * ib]);
        double Ecmi = fp->Ecm->at(pos + ie);
        double omega = fp->BQ_pointer->getOmegaFromElab(mu, Ecmi);
        // printf("DBUG vec: %d, Ecm = %e, omega = %e \t ret = %e\n", im, Ecmi, omega, ret);
        ret_r[pos + ie] = omega;
        ret += omega * omega;
      }
    }
  // printf("DBUG vec: %d, ret = %e\n", ib, ret);

  // printf("DBUG vec: ret = %e\n", ret);

  Eigen::MatrixXf cov_r_mat = Eigen::MatrixXf::Zero(energyNumAll, energyNumAll);

  double r_mean_of_blocks[energyNumAll] = {.0};

  for (size_t ilat = 0; ilat < lattNUM; ilat++)
    for (size_t ie = 0; ie < energyNumList[ilat]; ie++)
    {
      for (size_t ib = 0; ib < blockList[ilat]; ib++)
      {
        r_mean_of_blocks[offset_e[ilat] + ie] += (ret_r[offset_latt[ilat] + ib * energyNumList[ilat] + ie]) / blockList[ilat];
      }
      // r_mean_of_blocks[offset_e[ilat] + ie] /= blockList[ilat];
    }

  //calc cov_r

  for (size_t ilat = 0; ilat < lattNUM; ilat++)
    for (size_t irow = 0; irow < energyNumList[ilat]; irow++)
      for (size_t jcol = 0; jcol < energyNumList[ilat]; jcol++)
      {
        cov_r_mat(offset_e[ilat] + irow, offset_e[ilat] + jcol) = 0;
        size_t nblock = blockList[ilat];
        size_t ne = energyNumList[ilat];
        for (size_t ib = 0; ib < nblock; ib++)
        {
          cov_r_mat(offset_e[ilat] + irow, offset_e[ilat] + jcol) += (nblock - 1) * (ret_r[offset_latt[ilat] + ib * ne + irow] - r_mean_of_blocks[offset_e[ilat] + irow]) *
                                                                     (ret_r[offset_latt[ilat] + ib * ne + jcol] - r_mean_of_blocks[offset_e[ilat] + jcol]) / nblock;
        }
      }

  // printf("cov \n");
  // for (size_t irow = 0; irow < energyNumAll; irow++)
  // {
  // for (size_t icol = 0; icol < energyNumAll; icol++)
  // printf("%7.1e ", cov_r_mat(irow, icol));
  // printf("\n");
  // }
  // printf("\n\n\n");

  //calc chi^2
  cov_r_mat = cov_r_mat.inverse();

  // printf("cov inv \n");
  // for (size_t irow = 0; irow < energyNumAll; irow++)
  // {
  //   for (size_t icol = 0; icol < energyNumAll; icol++)
  //     printf("%7.1e ", cov_r_mat(irow, icol));
  //   printf("\n");
  // }
  // printf("\n\n\n");

  double chi2 = 0;
  for (size_t ilat = 0; ilat < lattNUM; ilat++)
    for (size_t irow = 0; irow < energyNumList[ilat]; irow++)
      for (size_t jcol = 0; jcol < energyNumList[ilat]; jcol++)
      {
        chi2 += r_mean_of_blocks[offset_e[ilat] + irow] * r_mean_of_blocks[offset_e[ilat] + jcol] * cov_r_mat(offset_e[ilat] + irow, offset_e[ilat] + jcol);
      }
  // exit(1);

  // return ret;
  return chi2;
}

int main()
{
  const std::string mDs_file = "/mnt/c/Users/shicj_laptop/Desktop/draw_Zc/DATA_jackknife/packed_2latt_Ds.npy";
  const std::string mD_file = "/mnt/c/Users/shicj_laptop/Desktop/draw_Zc/DATA_jackknife/packed_2latt_D.npy";
  const std::string mJPsi_file = "/mnt/c/Users/shicj_laptop/Desktop/draw_Zc/DATA_jackknife/packed_2latt_JPsi.npy";
  const std::string mPi_file = "/mnt/c/Users/shicj_laptop/Desktop/draw_Zc/DATA_jackknife/packed_2latt_Pi.npy";
  const std::string m12_file = "/mnt/c/Users/shicj_laptop/Desktop/draw_Zc/DATA_jackknife/packed_2latt_Zc.npy";
  const std::string L_mref_file = "/mnt/c/Users/shicj_laptop/Desktop/draw_Zc/DATA_jackknife/packed_2latt_L_mref.npy";

  std::vector<unsigned long> shape{};
  std::vector<double> m1A_data;
  std::vector<double> m1B_data;
  std::vector<double> m2A_data;
  std::vector<double> m2B_data;
  std::vector<double> m12_data;
  std::vector<double> L_mref_data;
  m1A_data = readNpy(mDs_file);
  m1B_data = readNpy(mD_file);
  m2A_data = readNpy(mJPsi_file);
  m2B_data = readNpy(mPi_file);
  m12_data = readNpy(m12_file);
  L_mref_data = readNpy(L_mref_file);
  if (m12_data.size() != globalSize || m12_data.size() != m1A_data.size() || m1A_data.size() != m2A_data.size())
  {
    throw(std::invalid_argument("Input  mass size not consistant!\n"));
  }

  vector<double> kpara;
  for (size_t i = 0; i < DEGREE_OF_FREEDOM; i++)
    kpara.push_back(.0);

  const size_t dof = DEGREE_OF_FREEDOM;

  const size_t npoints = 1000;
  double linespace[npoints];
  for (size_t i = 0; i < npoints; i++)
  {
    linespace[i] = (4.3 - 3.3) / 6.894 / npoints * i + 3.3 / 6.894;
  }

  // set<uint> powers_sumofpole;
  // powers_pole.insert(0);
  // powers.insert(1);
  // powers_sumofpole.insert(2);
  // powers.insert(3);
  set<uint> powers_poly;
  powers_poly.insert(0);
  // powers_poly.insert(1);
  powers_poly.insert(2);
  powers_poly.insert(3);
  powers_poly.insert(4);

  list<pair<KElementInfo, Polynomial>> pelems;
  pelems.push_back(make_pair(KElementInfo(2, 0, 2, 0, 0, 2, 0), Polynomial(powers_poly)));
  pelems.push_back(make_pair(KElementInfo(2, 0, 2, 1, 0, 2, 0), Polynomial(powers_poly)));
  pelems.push_back(make_pair(KElementInfo(2, 0, 2, 1, 0, 2, 1), Polynomial(powers_poly)));

  // list<pair<KElementInfo, SumOfPoles>> selems;
  // selems.push_back(make_pair(KElementInfo(2, 0, 2, 0, 0, 2, 0), SumOfPoles(powers_sumofpole)));
  // selems.push_back(make_pair(KElementInfo(2, 0, 2, 1, 0, 2, 0), SumOfPoles(powers_sumofpole)));
  // selems.push_back(make_pair(KElementInfo(2, 0, 2, 1, 0, 2, 1), SumOfPoles(powers_sumofpole)));

  // list<pair<KElementInfo, SumOfPolesPlusPolynomial>> spelems;
  // spelems.push_back(make_pair(KElementInfo(2, 0, 2, 0, 0, 2, 0), SumOfPolesPlusPolynomial(SumOfPoles(powers_sumofpole), Polynomial(powers_poly))));
  // spelems.push_back(make_pair(KElementInfo(2, 0, 2, 1, 0, 2, 0), SumOfPolesPlusPolynomial(SumOfPoles(powers_sumofpole), Polynomial(powers_poly))));
  // spelems.push_back(make_pair(KElementInfo(2, 0, 2, 1, 0, 2, 1), SumOfPolesPlusPolynomial(SumOfPoles(powers_sumofpole), Polynomial(powers_poly))));

  KtildeInverseCalculator Kinv(pelems);
  // KtildeMatrixCalculator K_mat(pelems, selems, spelems);
  cout << Kinv.output() << endl;
  cout << "Kmat #################################" << endl;

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

  double Elab_over_mref, Ecm, L_mref, m1, m2;
  uint chan = 0;
  cout.precision(15);

  // # input data array: L_mref(= a_t inv = xi times Ls), m12, m1A, m1B, m2A, m2B
  // # data shape for calc cov: shape = [data16, data24] = [[conf139, mom8], [conf144, mom14 ]]

  L_mref = 84.8;

  gsl_vector *x = gsl_vector_alloc(dof);
  gsl_vector *step_size;
  step_size = gsl_vector_alloc(dof);

  const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2rand;
  gsl_multimin_fminimizer *s = NULL;
  s = gsl_multimin_fminimizer_alloc(T, dof);
  int status;
  double size;

  BQ.setKtildeParameters(kpara);
  BQ.setRefMassL(L_mref);

  fit_params params = {
    BQ_pointer : &BQ,
    L_mref : &L_mref_data,
    Ecm : &m12_data,
    m1A : &m1A_data,
    m1B : &m1B_data,
    m2A : &m2A_data,
    m2B : &m2B_data,
  };
  gsl_multimin_function solverF;
  solverF.f = &my_minimaize_f;
  solverF.n = dof;
  solverF.params = &params;

  double old_chi2_res = .0;
  double new_chi2_res = .0;
  size_t iter = 0;
  do
  {
    if (iter == 0)
    {
      gsl_vector_set_all(x, .0);
      // gsl_vector_set(x, 0, 163.682);
      // gsl_vector_set(x, 1, -829.678);
      // gsl_vector_set(x, 2, -499.572);
      // gsl_vector_set(x, 3, 267.416);
      // gsl_vector_set(x, 4, -201.605);
      // gsl_vector_set(x, 5, 2268.153);
      // gsl_vector_set(x, 6, 110.245);
      // gsl_vector_set(x, 7, -2779.524);
      // gsl_vector_set(x, 8, -762.588);
      // gsl_vector_set(x, 9, 1427.861);
      // gsl_vector_set(x, 10, -288.597);
      // gsl_vector_set(x, 11, -470.220);
    }
    else
    {
      for (size_t i = 0; i < DEGREE_OF_FREEDOM; i++)
      {
        gsl_vector_set(x, i, gsl_vector_get(s->x, i));
      }
    }

    gsl_vector_set_all(step_size, 1e3);
    gsl_multimin_fminimizer_set(s, &solverF, x, step_size);

    printf("Start iter######################\n");

    do
    {
      iter++;
      status = gsl_multimin_fminimizer_iterate(s);
      if (status) /* check if solver is stuck */
        break;
      size = gsl_multimin_fminimizer_size(s);
      status = gsl_multimin_test_size(size, 1e-4);
      print_state(iter, s, size);
    } while (status == GSL_CONTINUE && iter < 1000000);

    print_state(iter, s, size);
    printf("status = %s\n", gsl_strerror(status));

    if (status == GSL_SUCCESS)
    {
      vector<double> Kinv_mat_save;
      Kinv_mat_save.reserve(1 * npoints * 4);
      vector<double> conv_kappa;
      for (size_t i = 0; i < DEGREE_OF_FREEDOM; i++)
        conv_kappa.push_back(gsl_vector_get(s->x, i));

      RealSymmetricMatrix Kinv_tosave_tmp;
      BQ.setKtildeParameters(conv_kappa);
      for (size_t ip = 0; ip < npoints; ip++)
      {
        BQ.getKtildeOrInverseFromEcm(linespace[ip], Kinv_tosave_tmp);
        for (int irow = 0; irow < 2; irow++)
          for (int icol = 0; icol < 2; icol++)
            Kinv_mat_save.push_back(Kinv_tosave_tmp.get(irow, icol));
      }
      const unsigned long save_shape[1] = {npoints * 4};
      if (Kinv_mat_save.size() != 1 * npoints * 4)
        throw(std::invalid_argument("Save vector size error!!\n"));
      else
        npy::SaveArrayAsNumpy("./savedata.npy", false, 1, save_shape, Kinv_mat_save);
      printf("Save: ./savedata.npy");
    }
    else
    {
      throw(std::invalid_argument("---------Fit failed!-------------------"));
    }

    new_chi2_res = s->fval;
    if (abs(new_chi2_res - old_chi2_res) < 1e-2)
    {
      old_chi2_res = new_chi2_res;
      break;
    }
    old_chi2_res = new_chi2_res;
  } while (true);

  gsl_vector_free(x);
  gsl_vector_free(step_size);
  gsl_multimin_fminimizer_free(s);

  return 0;
}

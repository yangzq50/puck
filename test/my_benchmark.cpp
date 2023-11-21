//
// Created by yzq on 11/16/23.
//
#define test_sift1M 0
//#define test_deep10M 0
//#define test_316 0
//#define test_1000 0

#include <cstring>
#include <iostream>
#include <memory>
#include "puck/index.h"
#include "test/test_index.h"
#include "puck/gflags/puck_gflags.h"
#include "puck/hierarchical_cluster/hierarchical_cluster_index.h"
#include "puck/puck/puck_index.h"
#include "puck/tinker/tinker_index.h"
#include "puck/puck/realtime_insert_puck_index.h"

constexpr bool build_tinker = true;
constexpr int dim = 96;
constexpr int nq = 10000;
constexpr int top_k = 100;
constexpr int repeat_time = 10;
constexpr size_t coarse_cluster_count = 100;
constexpr size_t fine_cluster_count = 100;
constexpr size_t search_coarse_count = 1;
constexpr size_t tinker_search_range = 100;
constexpr size_t train_points_count = 0;
constexpr char *sift1M_base = "/home/benchmark/benchmark_dataset/sift1M/sift_base.fvecs";
constexpr char *sift1M_query = "/home/benchmark/benchmark_dataset/sift1M/sift_query.fvecs";
constexpr char *sift1M_groundtruth = "/home/benchmark/benchmark_dataset/sift1M/sift_groundtruth.ivecs";
constexpr char *deep10M_base = "/home/benchmark/benchmark_dataset/deep10M/deep10M_base.fvecs";
constexpr char *deep10M_query = "/home/benchmark/benchmark_dataset/deep10M/deep10M_query.fvecs";
constexpr char *deep10M_groundtruth = "/home/benchmark/benchmark_dataset/deep10M/deep10M_groundtruth.ivecs";
#ifdef test_sift1M
constexpr char *choose_base = sift1M_base;
constexpr char *choose_query = sift1M_query;
constexpr char *choose_groundtruth = sift1M_groundtruth;
#endif
#ifdef test_deep10M
constexpr char *choose_base = deep10M_base;
constexpr char *choose_query = deep10M_query;
constexpr char *choose_groundtruth = deep10M_groundtruth;
#endif
constexpr char *index_path_root = "/home/benchmark/benchmark_save_index/tinker_index";
//sift1M save files
constexpr char *sift1M_feature_file_name = sift1M_base;
constexpr char *sift1M_train_fea_file_name = "sift1M/train_fea.dat";
constexpr char *sift1M_index_file_name = "sift1M/index.dat";
constexpr char *sift1M_coarse_codebook_file_name = "sift1M/coarse.dat";
constexpr char *sift1M_fine_codebook_file_name = "sift1M/fine.dat";
constexpr char *sift1M_cell_assign_file_name = "sift1M/cell_assign.dat";
constexpr char *sift1M_tinker_file_name = "sift1M/tinker_relations.dat";

//deep10M save files
constexpr char *deep10M_feature_file_name = deep10M_base;
constexpr char *deep10M_train_fea_file_name = "deep10M/train_fea.dat";
constexpr char *deep10M_index_file_name = "deep10M/index.dat";
constexpr char *deep10M_coarse_codebook_file_name = "deep10M/coarse.dat";
constexpr char *deep10M_fine_codebook_file_name = "deep10M/fine.dat";
constexpr char *deep10M_cell_assign_file_name = "deep10M/cell_assign.dat";
constexpr char *deep10M_tinker_file_name = "deep10M/tinker_relations.dat";

//choosen edition
#ifdef test_sift1M
constexpr char *feature_file_name = sift1M_feature_file_name;
constexpr char *train_fea_file_name = sift1M_train_fea_file_name;
constexpr char *index_file_name = sift1M_index_file_name;
constexpr char *coarse_codebook_file_name = sift1M_coarse_codebook_file_name;
constexpr char *fine_codebook_file_name = sift1M_fine_codebook_file_name;
constexpr char *cell_assign_file_name = sift1M_cell_assign_file_name;
constexpr char *tinker_file_name = sift1M_tinker_file_name;
#endif
#ifdef test_deep10M
constexpr char *feature_file_name = deep10M_feature_file_name;
constexpr char *train_fea_file_name = deep10M_train_fea_file_name;
constexpr char *index_file_name = deep10M_index_file_name;
constexpr char *coarse_codebook_file_name = deep10M_coarse_codebook_file_name;
constexpr char *fine_codebook_file_name = deep10M_fine_codebook_file_name;
constexpr char *cell_assign_file_name = deep10M_cell_assign_file_name;
constexpr char *tinker_file_name = deep10M_tinker_file_name;
#endif


double now_time() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

size_t physical_memory_used_by_process() {
    FILE *file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];

    while (fgets(line, 128, file) != nullptr) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            int len = strlen(line);

            const char *p = line;
            for (; std::isdigit(*p) == false; ++p) {
            }

            line[len - 3] = 0;
            result = atoi(p);

            break;
        }
    }

    fclose(file);

    return result;
}


class my_TestIndex {
public:
    my_TestIndex() {
        std::string feature_dim = std::to_string(dim);
        google::SetCommandLineOption("feature_dim", feature_dim.c_str());
        google::SetCommandLineOption("whether_norm", "false");
        auto s_search_coarse_count = std::to_string(search_coarse_count);
        google::SetCommandLineOption("search_coarse_count", s_search_coarse_count.c_str());
        auto s_tinker_search_range = std::to_string(tinker_search_range);
        google::SetCommandLineOption("tinker_search_range", s_tinker_search_range.c_str());
        //google::SetCommandLineOption("kmeans_iterations_count", "1");
        auto s_coarse_cluster_count = std::to_string(coarse_cluster_count);
        google::SetCommandLineOption("coarse_cluster_count", s_coarse_cluster_count.c_str());
        auto s_fine_cluster_count = std::to_string(fine_cluster_count);
        google::SetCommandLineOption("fine_cluster_count", s_fine_cluster_count.c_str());
        if (train_points_count != 0) {
            google::SetCommandLineOption("train_points_count", std::to_string(train_points_count).c_str());
        }
        _query_filename = choose_query;
        _groundtruth_filename = choose_groundtruth;
    }

    ~my_TestIndex() = default;

    int reset_index(int index_type) {
        if (index_type == int(puck::IndexType::TINKER)) {
            _index.reset(new puck::TinkerIndex());
        } else if (index_type == int(puck::IndexType::PUCK)) {
            _index.reset(new puck::PuckIndex());
        } else if (index_type == int(puck::IndexType::HIERARCHICAL_CLUSTER)) {
            _index.reset(new puck::HierarchicalClusterIndex());
        } else {
            std::cerr << "index type error.\n";
            return -1;
        }
        return 0;
    }

    int build_index(int index_type) {
        if (reset_index(index_type) != 0) {
            std::cerr << "reset index error.\n";
            return -1;
        }

        auto t0 = now_time();
        if (_index->train() != 0) {
            std::cerr << "train Fail.\n";
            return -1;
        }
        // output time for training
        auto t1 = now_time();
        std::cout << "time for training:" << t1 - t0 << std::endl;

        // output memory usage in MB
        std::cout << "after training, before building, consuming" << (physical_memory_used_by_process() / 1024.0) <<
                  " MB" << std::endl;
        if (_index->build() != 0) {
            std::cerr << "build Fail.\n";
            return -1;
        }
        auto t2 = now_time();
        std::cout << "time for building:" << t2 - t1 << std::endl;

        // output memory usage in MB
        std::cout << "after building, consuming" << (physical_memory_used_by_process() / 1024.0) <<
                  " MB" << std::endl;
        //_index.release();
        return 0;
    }

    //int insert_index(int thread_cnt);

    float cmp_search_recall() {
        if (_index->init() != 0) {
            std::cerr << "load index has Error" << std::endl;
            return -1;
        }
        std::cout << "load index suc." << std::endl; // output memory usage in MB
        std::cout << "after loading index, consuming" << (physical_memory_used_by_process() / 1024.0) << " MB" <<
                  std::endl;
        std::vector<float> query_feature(nq * dim);
        int ret = puck::read_fvec_format(_query_filename.c_str(), dim,
                                         nq, query_feature.data());

        //_groundtruth_filename
        if (ret != nq) {
            std::cerr << "load " << _query_filename << " has Error" << std::endl;
            return -1;
        }

        std::vector<std::vector<uint32_t>> groundtruth_data(nq);
        {
            std::ifstream input_file;
            input_file.open(_groundtruth_filename.c_str(), std::ios::binary);

            if (!input_file.good()) {
                input_file.close();
                std::cerr << "read all data file error : " << _groundtruth_filename << std::endl;
                return -1;
            }

            uint32_t d = 0;
            uint32_t i = 0;

            while (!input_file.eof() && i < nq) {
                input_file.read((char *) &d, sizeof(uint32_t));
                groundtruth_data[i].resize(d);
                input_file.read((char *) groundtruth_data[i].data(), sizeof(uint32_t) * d);
                ++i;
            }

            input_file.close();
        }

        puck::Request request;
        puck::Response response;
        request.topk = top_k;

        std::vector<float> distance(request.topk * nq);
        std::vector<uint32_t> local_idx(request.topk * nq);
        response.distance = distance.data();
        response.local_idx = local_idx.data();

        uint32_t match_pair_cnt_max = 0;
        for (int repeat_n = 0; repeat_n < repeat_time; ++repeat_n) {
            uint32_t match_pair_cnt = 0;
            double tot_time = 0;
            double calc_recall_time = 0;

            for (int i = 0; i < nq; ++i) {
                request.feature = query_feature.data() + i * dim;

                auto t0 = now_time();
                ret = _index->search(&request, &response);
                tot_time += now_time() - t0;
                auto t1 = now_time();

                if (ret != 0) {
                    std::cerr << "search item " << i << " error" << ret << std::endl;
                    break;
                }

                for (int j = 0; j < 100 && j < (int) response.result_num; j++) {
                    //LOG(INFO)<<"\t"<<i<<"\t"<<j<<"\t"<<response.local_idx[j]<<" "<<groundtruth_data[i][j];

                    auto ite = std::find(groundtruth_data[i].begin(), groundtruth_data[i].end(), response.local_idx[j]);

                    if (ite != groundtruth_data[i].end()) {
                        ++match_pair_cnt;
                    }
                }
                calc_recall_time += now_time() - t1;
            }
            if (repeat_n == 0) {
                std::cout << "\n###########################################" << std::endl;
                double avg_tot_cnt = (double) response.tot_cnt / nq;
                std::cout << "avg_tot_cnt:" << avg_tot_cnt << std::endl;
                double avg_loop_cnt = (double) response.loop_cnt / nq;
                std::cout << "avg_loop_cnt:" << avg_loop_cnt << std::endl;
                //output percentage
                std::cout << "percentage:" << avg_loop_cnt / avg_tot_cnt << std::endl;
                //output avg hnsw distance_computations
                std::cout << "avg hnsw distance_computations:"
                          << (double) (((puck::TinkerIndex *) _index.get())->_tinker_index->distance_computations_) / nq
                          << std::endl;
                std::cout << "\n###########################################\n\n" << std::endl;
            }
            std::cout << "\n###########################################" << std::endl;
            //output search_coarse_count
            std::cout << "search_coarse_count:" << puck::FLAGS_search_coarse_count << std::endl;
            //output tinker_search_range
            std::cout << "tinker_search_range:" << puck::FLAGS_tinker_search_range << std::endl;
            std::cout << "calc recall time: " << calc_recall_time << " s" << std::endl;
            //output search time
            std::cout << "\n###########################################" << std::endl;
            std::cout << "time: " << tot_time << " s" << std::endl;
            //output qps
            std::cout << "qps: " << nq / (tot_time) << std::endl;
            std::cout << "recall = " << 1.0 * match_pair_cnt / (nq * top_k) << std::endl;
            std::cout << "###########################################\n" << std::endl;
            match_pair_cnt_max = std::max(match_pair_cnt_max, match_pair_cnt);
        }
        return 1.0 * match_pair_cnt_max / (nq * top_k);
    }

    void release() {
        _index.reset(nullptr);
    }

private:
    std::unique_ptr<puck::Index> _index;
    std::string _query_filename;
    std::string _groundtruth_filename;
};

int main() {
    google::SetCommandLineOption("index_path", index_path_root);
    google::SetCommandLineOption("feature_file_name", feature_file_name);
    google::SetCommandLineOption("train_fea_file_name", train_fea_file_name);
    google::SetCommandLineOption("index_file_name", index_file_name);
    google::SetCommandLineOption("coarse_codebook_file_name", coarse_codebook_file_name);
    google::SetCommandLineOption("fine_codebook_file_name", fine_codebook_file_name);
    google::SetCommandLineOption("cell_assign_file_name", cell_assign_file_name);
    google::SetCommandLineOption("tinker_file_name", tinker_file_name);
    google::SetCommandLineOption("train_fea_file_name", train_fea_file_name);
    auto t_out = now_time();
    auto kb_ = physical_memory_used_by_process();
    std::cout << "begin, consuming" << kb_ / 1024.0 << " MB" << std::endl;
    {
        my_TestIndex ti;
        auto kb_0 = physical_memory_used_by_process();
        std::cout << "Before test, consuming" << kb_0 / 1024.0 << " MB" << std::endl;
        {
            auto t0 = now_time();
            ti.release();
            auto kb_r = physical_memory_used_by_process();
            std::cout << "after releasing old index, consuming" << kb_r / 1024.0 << " MB" << std::endl;
            auto t1 = now_time();
            std::cout << "time passed:" << t1 - t0 << std::endl;
            auto t2 = t1;
            if (build_tinker) {
                ti.build_index(int(puck::IndexType::TINKER));
                auto kb_5 = physical_memory_used_by_process();
                std::cout << "after build TinkerIndex, consuming" << kb_5 / 1024.0 << " MB" << std::endl;
                t2 = now_time();
                std::cout << "time passed:" << t2 - t1 << std::endl;
            }
            auto rec3 = ti.cmp_search_recall();
            std::cout << "recall=" << rec3 << std::endl;
            auto kb_6 = physical_memory_used_by_process();
            std::cout << "after calculate TinkerIndex recall, consuming" << kb_6 / 1024.0 << " MB" << std::endl;
            auto t3 = now_time();
            std::cout << "time passed:" << t3 - t2 << std::endl;
        }
        auto t_out3 = now_time();
        std::cout << "\n\nTinkerIndex, time passed:" << t_out3 - t_out << "\n\n";
    }
    auto kb_6 = physical_memory_used_by_process();
    std::cout << "finished, consuming" << kb_6 / 1024.0 << " MB" << std::endl;
    std::cout << "total time passed:" << now_time() - t_out << std::endl;
}

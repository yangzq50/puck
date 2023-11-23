//
// Created by yzq on 11/16/23.
//

#include <filesystem>
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

int dim = 96;
int nq = 10000;
int top_k = 100;
int repeat_time = 10;
size_t coarse_cluster_count = 100;
size_t fine_cluster_count = 100;
size_t tinker_construction = 200;
size_t search_coarse_count = 1;
size_t tinker_search_range = 100;
size_t train_points_count = 0;

const char *choose_query = nullptr;
const char *choose_groundtruth = nullptr;

const char *sift1M_query = "/home/benchmark/benchmark_dataset/sift1M/sift_query.fvecs";
const char *sift1M_groundtruth = "/home/benchmark/benchmark_dataset/sift1M/sift_groundtruth.ivecs";

const char *deep10M_query = "/home/benchmark/benchmark_dataset/deep10M/deep10M_query.fvecs";
const char *deep10M_groundtruth = "/home/benchmark/benchmark_dataset/deep10M/deep10M_groundtruth.ivecs";

const char *index_path_root = "/home/benchmark/benchmark_save_index/tinker_index";

//sift1M base files
const char *sift1M_feature_file_name = "sift1M_base.fvecs";

//deep10M base files
const char *deep10M_feature_file_name = "deep10M_base.fvecs";



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
    std::string feature_file_name;
    std::string dataset_subdir;
    std::string train_fea_file_name = "train_fea.dat";
    std::string index_file_name = "index.dat";
    std::string coarse_codebook_file_name = "coarse.dat";
    std::string fine_codebook_file_name = "fine.dat";
    std::string cell_assign_file_name = "cell_assign.dat";
    std::string tinker_file_name = "tinker_relations.dat";
    //let user choose to test sift1M or deep10M
    enum test_type {
        invalid = 0, sift1M = 1, deep10M = 2
    };
    std::cout << "please choose to test sift1M or deep10M, input 1 or 2" << std::endl;
    std::cout << "1. sift1M" << std::endl;
    std::cout << "2. deep10M" << std::endl;
    int choose = 0;
    std::cin >> choose;
    switch (choose) {
        case test_type::sift1M: {
            dataset_subdir = "sift1M/";
            dim = 128;
            nq = 10000;
            choose_query = sift1M_query;
            choose_groundtruth = sift1M_groundtruth;
            feature_file_name = sift1M_feature_file_name;
            break;
        }
        case test_type::deep10M: {
            dataset_subdir = "deep10M/";
            dim = 96;
            nq = 10000;
            choose_query = deep10M_query;
            choose_groundtruth = deep10M_groundtruth;
            feature_file_name = deep10M_feature_file_name;
            break;
        }
        default:
            std::cout << "invalid input, exit." << std::endl;
            return -1;
    }
    {
        std::cout << "Choose some parameters, or input 0 to use default values." << std::endl;
        //let user choose top_k and repeat_time
        std::cout << "please input top_k and repeat_time, default values are 100 and 10." << std::endl;
        int input_top_k = 0, input_repeat_time = 0;
        std::cin >> input_top_k >> input_repeat_time;
        if (input_top_k != 0) {
            top_k = input_top_k;
        }
        if (input_repeat_time != 0) {
            repeat_time = input_repeat_time;
        }
        //let user choose tinker_construction
        std::cout << "please input tinker_construction, default value is 200." << std::endl;
        int input_tinker_construction = 0;
        std::cin >> input_tinker_construction;
        if (input_tinker_construction != 0) {
            tinker_construction = input_tinker_construction;
        }
        //let user choose coarse_cluster_count and fine_cluster_count
        std::cout << "please input coarse_cluster_count and fine_cluster_count, default values are 100 and 100."
                  << std::endl;
        size_t input_coarse_cluster_count = 0, input_fine_cluster_count = 0;
        std::cin >> input_coarse_cluster_count >> input_fine_cluster_count;
        if (input_coarse_cluster_count != 0) {
            coarse_cluster_count = input_coarse_cluster_count;
        }
        if (input_fine_cluster_count != 0) {
            fine_cluster_count = input_fine_cluster_count;
        }
        //let user choose search_coarse_count and tinker_search_range
        std::cout << "please input search_coarse_count and tinker_search_range, default values are 1 and 100."
                  << std::endl;
        size_t input_search_coarse_count = 0, input_tinker_search_range = 0;
        std::cin >> input_search_coarse_count >> input_tinker_search_range;
        if (input_search_coarse_count != 0) {
            search_coarse_count = input_search_coarse_count;
        }
        if (input_tinker_search_range != 0) {
            tinker_search_range = input_tinker_search_range;
        }
        //let user choose train_points_count
        std::cout << "please input train_points_count." << std::endl;
        std::cin >> train_points_count;
        if (train_points_count != 0) {
            google::SetCommandLineOption("train_points_count", std::to_string(train_points_count).c_str());
        }
    }
    {
        //output chosen parameters
        std::cout << "chosen parameters:" << std::endl;
        std::cout << "train_points_count=" << puck::FLAGS_train_points_count << std::endl;
        std::cout << "dim=" << dim << std::endl;
        std::cout << "nq=" << nq << std::endl;
        std::cout << "top_k=" << top_k << std::endl;
        std::cout << "repeat_time=" << repeat_time << std::endl;
        std::cout << "tinker_construction=" << puck::FLAGS_tinker_construction << std::endl;
        std::cout << "coarse_cluster_count=" << puck::FLAGS_coarse_cluster_count << std::endl;
        std::cout << "fine_cluster_count=" << puck::FLAGS_fine_cluster_count << std::endl;
        std::cout << "search_coarse_count=" << puck::FLAGS_search_coarse_count << std::endl;
        std::cout << "tinker_search_range=" << puck::FLAGS_tinker_search_range << std::endl;
    }
    {
        std::string feature_dim = std::to_string(dim);
        google::SetCommandLineOption("feature_dim", feature_dim.c_str());
        google::SetCommandLineOption("whether_norm", "false");
        auto s_search_coarse_count = std::to_string(search_coarse_count);
        google::SetCommandLineOption("search_coarse_count", s_search_coarse_count.c_str());
        auto s_tinker_search_range = std::to_string(tinker_search_range);
        google::SetCommandLineOption("tinker_search_range", s_tinker_search_range.c_str());
        //google::SetCommandLineOption("kmeans_iterations_count", "1");
        auto s_tinker_construction = std::to_string(tinker_construction);
        google::SetCommandLineOption("tinker_construction", s_tinker_construction.c_str());
        auto s_coarse_cluster_count = std::to_string(coarse_cluster_count);
        google::SetCommandLineOption("coarse_cluster_count", s_coarse_cluster_count.c_str());
        auto s_fine_cluster_count = std::to_string(fine_cluster_count);
        google::SetCommandLineOption("fine_cluster_count", s_fine_cluster_count.c_str());
        std::string file_suffix = ".tinker_construction." + s_tinker_construction +
                                  ".coarse_cluster_count." + s_coarse_cluster_count + ".fine_cluster_count." +
                                  s_fine_cluster_count;

        google::SetCommandLineOption("index_path", index_path_root);
        google::SetCommandLineOption("feature_file_name", feature_file_name.c_str());
        std::string real_train_fea_file_name = dataset_subdir + train_fea_file_name + file_suffix;
        google::SetCommandLineOption("train_fea_file_name", real_train_fea_file_name.c_str());
        std::string real_index_file_name = dataset_subdir + index_file_name + file_suffix;
        google::SetCommandLineOption("index_file_name", real_index_file_name.c_str());
        std::string real_coarse_codebook_file_name = dataset_subdir + coarse_codebook_file_name + file_suffix;
        google::SetCommandLineOption("coarse_codebook_file_name", real_coarse_codebook_file_name.c_str());
        std::string real_fine_codebook_file_name = dataset_subdir + fine_codebook_file_name + file_suffix;
        google::SetCommandLineOption("fine_codebook_file_name", real_fine_codebook_file_name.c_str());
        std::string real_cell_assign_file_name = dataset_subdir + cell_assign_file_name + file_suffix;
        google::SetCommandLineOption("cell_assign_file_name", real_cell_assign_file_name.c_str());
        std::string real_tinker_file_name = dataset_subdir + tinker_file_name + file_suffix;
        google::SetCommandLineOption("tinker_file_name", real_tinker_file_name.c_str());
    }

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
            {
                //build index if it doesn't exist
                std::string tinker_file_path = puck::FLAGS_index_path + "/" + puck::FLAGS_tinker_file_name;
                if (!std::filesystem::exists(tinker_file_path)) {
                    ti.build_index(int(puck::IndexType::TINKER));
                    auto kb_5 = physical_memory_used_by_process();
                    std::cout << "after build TinkerIndex, consuming" << kb_5 / 1024.0 << " MB" << std::endl;
                    t2 = now_time();
                    std::cout << "time passed:" << t2 - t1 << std::endl;
                }
            }
            ti.reset_index(int(puck::IndexType::TINKER));
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

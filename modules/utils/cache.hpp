#pragma once

#include <boost/bimap.hpp> 
#include <boost/bimap/list_of.hpp> 
#include <boost/bimap/set_of.hpp> 
#include <boost/function.hpp> 

#include "config.hpp"
#include "misc.hpp"
#include "numerics.hpp"
#include "cycletimer.hpp"

#if ENABLE_MULTITHREADING && ENABLE_OPENMP
#include <omp.h>
#endif

/// This header file implements multiple versions of an LRU cache.
///
/// SingleCache is the basic cache implementation, where all threads
/// are expected to access a single cache.
///
/// MultiCache implements 1 SingleCache for each thread, avoiding the 
/// need for locking in each of the Caches.
///
/// MultiRingCache implements 1 SingleCache for each thread, however, each
/// cache is responsible for a sequential set of key values (instead of
/// each cache being responsible for a thread) arranged in a ring.
/// However, each SingleCache must be lock protected, since multiple threads 
/// may map their keys to the same cache.
///
/// MultiRingPriorityCache implements the same thing as the MultiRingCache
/// except there are no locks within each SingleCache.  Instead the lock
/// exists within the ring such that a thread will try to choose the cache
/// that has been assigned to the key value if possible, otherwise it will just
/// query the next non locked cache.


/// Defines an LRU cache.
/// If B is true, this class is thread safe, else it is not thread safe.
/// K is the key type, V is the value type, and SET is the SET storage type, ex. boost::bimap::set_of
template <bool B, typename K, typename V> 
class SingleCache { 
 public: 
  typedef boost::bimaps::bimap<boost::bimaps::set_of<K>, boost::bimaps::list_of<V> > container_type; 
 
  // Constuctor specifies the cached function and 
  // the maximum number of records to be stored. 
  SingleCache(const boost::function<V(const K&)>& f, size_t c) : _fn(f), _capacity(c) { 
    _hits = 0;
    _misses = 0;
  } 
 
  // Non locking version
  template<typename U = V> typename std::enable_if<!B, U>::type operator()(const K& k) { 
    SCOPED_TIMER

    const typename container_type::left_iterator it = _container.left.find(k); 
    if (it == _container.left.end()) {      
      V v = _fn(k);
      insert(k,v); 
      _misses++;
      return v;
    } else {
      _container.right.relocate(_container.right.end(), _container.project_right(it)); 
      _hits++;
      return it->second;
    }

  } 

  template<typename U = std::vector<V> > typename std::enable_if<!B, U>::type operator()(const std::vector<K> &k) { 
    U u(k.size());
    for(size_t i=0; i<k.size(); i++) {
      u[i] = (*this)(k);
    }
    return u;
  } 

  // Locking version
  template<typename U = V> typename std::enable_if<B, U>::type operator()(const K& k) { 
    SCOPED_TIMER
    
    V v;
    #pragma omp critical
    {
      const typename container_type::left_iterator it = _container.left.find(k); 
      if (it == _container.left.end()) {
        v = _fn(k);
        insert(k,v); 
        // _misses++;
      } else {
        _container.right.relocate(_container.right.end(), _container.project_right(it)); 
        v = it->second;
        // _hits++;
      }
    }
    return v;
  }

  template<typename U = std::vector<V> > typename std::enable_if<B, U>::type operator()(const std::vector<K> &k) { 
    U u(k.size());
    for(size_t i=0; i<k.size(); i++) {
      u[i] = (*this)(k);
    }
    return u;
  } 

  uint64_t capacity()         const { return _capacity; }
  // uint64_t hits() const { return _hits; }
  // uint64_t misses() const { return _misses; }
  
 private: 
  void insert(const K& k, const V& v) { 
    if (_container.size() == _capacity) { 
      _container.right.erase(_container.right.begin()); 
    }  
    _container.insert(typename container_type::value_type(k,v)); 
  } 
 
  const boost::function<V(const K&)> _fn; 
  const size_t _capacity; 
  container_type _container; 
  uint64_t _hits, _misses;
}; 

#if ENABLE_MULTITHREADING && ENABLE_OPENMP
/// Implements a SingleCache for each OpenMP thread.
template <typename K, typename V> 
class MultiCache { 
 public: 

  MultiCache(const boost::function<V(const K&)>& f, size_t c) : _fn(f), _capacity(c) { 
    for(int i=0; i < omp_get_max_threads(); i++){
      _caches.push_back(PTR_LIB::make_shared<SingleCache<false, K, V>>(f, c / omp_get_max_threads()));
    }   
  } 
 
  // Obtain value of the cached function for k 
  V operator()(const K& k) { 
    SCOPED_TIMER
    return (*_caches[omp_get_thread_num()])(k);
  } 

  uint64_t capacity() const { return _capacity; }
  
 private: 

  std::vector< PTR_LIB::shared_ptr<SingleCache<false, K, V> > > _caches;
  const boost::function<V(const K&)> _fn; 
  const size_t _capacity; 
}; 

/// Implements a SingleCache for each OpenMP thread, where each cache is responsible
/// for sequential ranges of items, this requires that K has modulo operator % and a relation operator <.
/// This is essentially a bunch of caches arranged in a ring topology.
template <typename K, typename V> 
class MultiRingCache { 
 public: 

  MultiRingCache(const boost::function<V(const K&)>& f, size_t c) : _fn(f), _capacity(c), _single_capacity(c / omp_get_max_threads()) { 
    for(int i=0; i < omp_get_max_threads(); i++){
      _caches.push_back(PTR_LIB::make_shared<SingleCache<false, K, V>>(f, _single_capacity));
    }   
  } 
 
  // Obtain value of the cached function for k 
  V operator()(const K& k) { 
    SCOPED_TIMER
    return (*_caches[(size_t)(k / _single_capacity) % omp_get_max_threads()])(k);
  } 

  uint64_t capacity() const { return _capacity; }
  
 private: 
  std::vector< PTR_LIB::shared_ptr<SingleCache<false, K, V> > > _caches;
  const boost::function<V(const K&)> _fn; 
  const size_t _capacity, _single_capacity; 
}; 


/// Implements a SingleCache for each OpenMP thread, where each cache is responsible
/// for sequential ranges of items, this requires that K has modulo operator % and a relation operator <.
/// This is essentially a bunch of caches arranged in a ring topology.
template <typename K, typename V> 
class MultiRingPriorityCache { 
 public: 

  MultiRingPriorityCache(const boost::function<V(const K&)>& f, size_t c) : _fn(f), _capacity(c), 
    _single_capacity(c / omp_get_max_threads()) { 
    
    _locks.resize(omp_get_max_threads());
    for(int i=0; i < omp_get_max_threads(); i++){
      _caches.push_back(PTR_LIB::make_shared<SingleCache<false, K, V>>(f, _single_capacity));
      omp_init_lock(&_locks[i]);
    }   
  } 
 
  // Obtain value of the cached function for k 
  V operator()(const K& k) { 
    SCOPED_TIMER
    // double startlookup = CycleTimer::currentSeconds();
    for(size_t i = 0; i<omp_get_max_threads(); i++) {
      size_t cache_idx = ((size_t)(k / _single_capacity)+i) % omp_get_max_threads();
      int lock_acquired = omp_test_lock(&_locks[cache_idx]);
      if(lock_acquired) {
        V v = (*_caches[cache_idx])(k);
        omp_unset_lock(&_locks[cache_idx]);
        return v;
      }
    }

    size_t cache_idx = ((size_t)(k / _single_capacity)) % omp_get_max_threads();
    omp_set_lock(&_locks[cache_idx]);
    V v = (*_caches[cache_idx])(k);
    omp_unset_lock(&_locks[cache_idx]);

    return v;
  } 

  uint64_t capacity() const { return _capacity; }
  
 private: 

  std::vector< PTR_LIB::shared_ptr<SingleCache<false, K, V> > > _caches;
  std::vector<omp_lock_t> _locks;
  
  const boost::function<V(const K&)> _fn; 
  const size_t _capacity, _single_capacity; 
  double _lookup_time_total;
}; 

typedef MultiRingPriorityCache<uint64_t, numerics::sparse_vector_t> bow_ring_priority_cache_t;
typedef MultiRingCache<uint64_t, numerics::sparse_vector_t> bow_ring_cache_t;
typedef MultiCache<uint64_t, numerics::sparse_vector_t> bow_multi_cache_t;

typedef MultiRingPriorityCache<uint64_t,  std::vector<float>> vec_ring_priority_cache_t;
typedef MultiRingCache<uint64_t,  std::vector<float>> vec_ring_cache_t;
typedef MultiCache<uint64_t, std::vector<float>> vec_multi_cache_t;

#endif

typedef SingleCache<true, uint64_t, numerics::sparse_vector_t> bow_single_cache_t;
typedef SingleCache<true, uint64_t, std::vector<float>> vec_single_cache_t;


import { useState } from 'react'
import { Search as SearchIcon, FileText, Image, Sparkles } from 'lucide-react'
import apiClient from '../api/client'
import { Input } from '@/components/ui/input'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

interface SearchResult {
  chunk_id: string
  doc_id: string
  kb_id: string
  document_name: string
  content: string
  chunk_type: string
  page_number: number | null
  score: number
  image_url: string
  highlight: string
}

export default function Search() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [searched, setSearched] = useState(false)
  const [elapsedMs, setElapsedMs] = useState(0)

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return
    setLoading(true)
    const start = Date.now()
    try {
      const res = await apiClient.post('/search', {
        query,
        top_k: 10,
        search_mode: 'hybrid',
      })
      setResults(res.data.results || [])
      setElapsedMs(Date.now() - start)
      setSearched(true)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="p-8 max-w-4xl mx-auto">
      {/* 头部 */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
          <Sparkles size={24} className="text-blue-500" />
          全局检索
        </h1>
        <p className="text-sm text-gray-500 mt-1">跨知识库的多模态混合检索</p>
      </div>

      {/* 搜索框 */}
      <form onSubmit={handleSearch} className="flex gap-2 mb-8">
        <div className="flex-1 relative">
          <SearchIcon size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="输入检索内容，支持多语言..."
            className="w-full h-12 pl-10 pr-4 border border-gray-300 rounded-xl text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>
        <button
          type="submit"
          disabled={loading || !query.trim()}
          className="h-12 px-6 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 disabled:opacity-50 transition"
        >
          {loading ? '检索中...' : '检索'}
        </button>
      </form>

      {/* 结果统计 */}
      {searched && !loading && (
        <p className="text-sm text-gray-500 mb-4">
          找到 {results.length} 个结果，耗时 {elapsedMs}ms
        </p>
      )}

      {/* 结果列表 */}
      <div className="space-y-3">
        {loading && (
          <div className="text-center py-12 text-gray-400">检索中...</div>
        )}

        {!loading && searched && results.length === 0 && (
          <div className="text-center py-12 text-gray-400">
            <SearchIcon size={40} className="mx-auto mb-3 opacity-50" />
            <p>未找到相关内容</p>
          </div>
        )}

        {results.map((r, i) => (
          <Card key={r.chunk_id || i} className="hover:shadow-sm transition-shadow">
            <CardContent className="p-5">
              <div className="flex items-center gap-2 mb-2">
                <FileText size={14} className="text-gray-400" />
                <span className="text-sm font-medium text-gray-700">
                  {r.document_name}
                </span>
                {r.page_number && (
                  <span className="text-xs text-gray-400">第 {r.page_number} 页</span>
                )}
                <Badge variant="secondary" className="ml-auto text-xs">
                  {(r.score * 100).toFixed(1)}%
                </Badge>
              </div>
              <p className="text-sm text-gray-600 leading-relaxed">
                {r.highlight || r.content}
              </p>
              {r.image_url && (
                <div className="mt-2 flex items-center gap-1 text-xs text-blue-500">
                  <Image size={12} />
                  <span>包含图片</span>
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}

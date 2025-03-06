// index.js
const express = require('express');
const axios = require('axios');
require('dotenv').config(); // 환경 변수 로드

const app = express();
app.use(express.json()); // JSON 요청 파싱

const GITHUB_TOKEN = process.env.GITHUB_TOKEN;
const GITHUB_USERNAME = 'YangTaeUk';
const DEFAULT_BRANCH = 'main'; // 또는 'master'
// 신규 레포지터리 생성 API 엔드포인트
app.post('/createRepo', async (req, res) => {
  const { repoName, description, isPrivate } = req.body;

  // GitHub API에 전달할 데이터 형식
  const data = {
    name: repoName,
    description: description || '',
    private: isPrivate || false,
  };

  try {
    // GitHub API 호출: /user/repos 엔드포인트에 POST 요청
    const response = await axios.post(
      'https://api.github.com/user/repos',
      data,
      {
        headers: {
          'Authorization': `token ${GITHUB_TOKEN}`,
          'Accept': 'application/vnd.github+json'
        }
      }
    );

    // 성공 시, 생성된 레포지터리 정보를 클라이언트에 반환
    res.status(201).json({
      success: true,
      repo: response.data
    });
  } catch (error) {
    // 오류 처리: 오류 메시지 및 상태 코드 반환
    console.error('Repository creation error:', error.response?.data || error.message);
    res.status(500).json({
      success: false,
      error: error.response?.data || error.message
    });
  }
});

app.post('/commitFile', async (req, res) => {
  const { repoName, filePath, fileContent, commitMessage } = req.body;
  const url = `https://api.github.com/repos/${GITHUB_USERNAME}/${repoName}/contents/${filePath}`;

  try {
    const response = await axios.put(url, {
      message: commitMessage,
      content: Buffer.from(fileContent).toString('base64')  // 파일 내용을 Base64 인코딩
    }, {
      headers: {
        'Authorization': `token ${GITHUB_TOKEN}`,
        'Accept': 'application/vnd.github+json'
      }
    });

    res.status(201).json({ success: true, data: response.data });
  } catch (error) {
    console.error('Error creating file:', error.response ? error.response.data : error.message);
    res.status(500).json({ success: false, error: error.response ? error.response.data : error.message });
  }
});

// 파일 커밋 및 푸시 API 엔드포인트
// app.post('/commitFile', async (req, res) => {
//   const { repoName, filePath, fileContent, commitMessage } = req.body;
//
//   try {
//     // 1. 기본 브랜치의 최신 커밋 SHA 조회
//     const refUrl = `https://api.github.com/repos/${GITHUB_USERNAME}/${repoName}/git/refs/heads/${DEFAULT_BRANCH}`;
//     const refResponse = await axios.get(refUrl, {
//       headers: {
//         'Authorization': `token ${GITHUB_TOKEN}`,
//         'Accept': 'application/vnd.github+json'
//       }
//     });
//     const latestCommitSha = refResponse.data.object.sha;
//
//     // 2. 최신 커밋의 트리 SHA 가져오기
//     const commitUrl = `https://api.github.com/repos/${GITHUB_USERNAME}/${repoName}/git/commits/${latestCommitSha}`;
//     const commitResponse = await axios.get(commitUrl, {
//       headers: {
//         'Authorization': `token ${GITHUB_TOKEN}`,
//         'Accept': 'application/vnd.github+json'
//       }
//     });
//     const baseTreeSha = commitResponse.data.tree.sha;
//
//     // 3. 파일 내용을 Blob으로 생성
//     const blobUrl = `https://api.github.com/repos/${GITHUB_USERNAME}/${repoName}/git/blobs`;
//     const blobResponse = await axios.post(blobUrl, {
//       content: fileContent,
//       encoding: 'utf-8'
//     }, {
//       headers: {
//         'Authorization': `token ${GITHUB_TOKEN}`,
//         'Accept': 'application/vnd.github+json'
//       }
//     });
//     const blobSha = blobResponse.data.sha;
//
//     // 4. 새로운 Tree 생성 (기존 트리를 기반으로 새 파일 추가)
//     const treeUrl = `https://api.github.com/repos/${GITHUB_USERNAME}/${repoName}/git/trees`;
//     const treeResponse = await axios.post(treeUrl, {
//       base_tree: baseTreeSha,
//       tree: [
//         {
//           path: filePath,   // 예: "folder/newfile.txt"
//           mode: '100644',
//           type: 'blob',
//           sha: blobSha
//         }
//       ]
//     }, {
//       headers: {
//         'Authorization': `token ${GITHUB_TOKEN}`,
//         'Accept': 'application/vnd.github+json'
//       }
//     });
//     const newTreeSha = treeResponse.data.sha;
//
//     // 5. 새 트리를 참조하는 새로운 Commit 생성
//     const commitPostUrl = `https://api.github.com/repos/${GITHUB_USERNAME}/${repoName}/git/commits`;
//     const newCommitResponse = await axios.post(commitPostUrl, {
//       message: commitMessage,
//       tree: newTreeSha,
//       parents: [latestCommitSha]
//     }, {
//       headers: {
//         'Authorization': `token ${GITHUB_TOKEN}`,
//         'Accept': 'application/vnd.github+json'
//       }
//     });
//     const newCommitSha = newCommitResponse.data.sha;
//
//     // 6. 브랜치 참조 업데이트 (새 커밋으로 "푸시")
//     const updateRefUrl = `https://api.github.com/repos/${GITHUB_USERNAME}/${repoName}/git/refs/heads/${DEFAULT_BRANCH}`;
//     await axios.patch(updateRefUrl, {
//       sha: newCommitSha
//     }, {
//       headers: {
//         'Authorization': `token ${GITHUB_TOKEN}`,
//         'Accept': 'application/vnd.github+json'
//       }
//     });
//
//     res.status(201).json({ success: true, commit: newCommitSha });
//   } catch (error) {
//     console.error('Commit error:', error.response ? error.response.data : error.message);
//     res.status(500).json({ success: false, error: error.response ? error.response.data : error.message });
//   }
// });

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
